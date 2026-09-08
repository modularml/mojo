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

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

namespace {
constexpr llvm::StringLiteral kStructElemLayoutError =
    "failed to get size/alignment for struct element type";
} // namespace

//===----------------------------------------------------------------------===//
// ArgConvention
//===----------------------------------------------------------------------===//

/// Return a string like "imm" or "mut".
const char *KGEN::getUserSyntax(ArgConvention convention) {
  switch (convention) {
  case ArgConvention::ImmReg:
  case ArgConvention::ImmMem:
    return "imm";
  case ArgConvention::OwnedReg:
  case ArgConvention::OwnedMem:
    return "var";
  case ArgConvention::DeinitMem:
    return "deinit";
  case ArgConvention::Mut:
    return "mut";
  case ArgConvention::Ref:
    return "ref";
  case ArgConvention::MutRef:
    return "ref";
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    return "out";
  }
  llvm_unreachable("invalid convention");
}

//===----------------------------------------------------------------------===//
// KGENDialect
//===----------------------------------------------------------------------===//

void KGENDialect::registerTypes() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "Mojo/KGENDialect/KGENTypes.cpp.inc"
      >();

  // Register custom type parser and printers for KGEN types.
  registerPrettyType(
      "non_struct_type", &NonStructTypeType::parse,
      mlir::TypeID::get<NonStructTypeType>(),
      +[](AsmPrinter &p, Type) { p << "non_struct_type"; });
  registerPrettyType(
      "type", &TypeType::parse, mlir::TypeID::get<TypeType>(),
      +[](AsmPrinter &p, Type) { p << "type"; });
  registerMnemonicType<DTypeType>();
  registerMnemonicType<PointerType>();
  registerMnemonicType<NoneType>();
  registerMnemonicType<StringType>();
  registerMnemonicType<ParamListType>();
  registerMnemonicType<TargetType>();
  registerMnemonicType<BuildInfoType>();
  registerMnemonicType<StructType>();
  registerMnemonicType<StructInstanceType>();
  registerMnemonicType<TypeValueType>();
  registerMnemonicType<VariantType>();

  // Register the SIMD type with custom parser/printer that sugars
  // `!kgen.simd<1, dtype>` to `!kgen.scalar<dtype>`.
  registerPrettyType(
      "simd", &SIMDType::parse, mlir::TypeID::get<SIMDType>(),
      +[](AsmPrinter &p, Type type) {
        auto simd = cast<SIMDType>(type);
        if (simd.isScalar()) {
          p << "scalar<";
          printDTypeParamValue(p, simd.getDType());
          p << ">";
        } else {
          p << "simd";
          simd.print(p);
        }
      });
  registerKeywordParser("scalar", [](AsmParser &p) -> Type {
    TypedAttr resultDType;
    // Parse literal '<' + dtype + literal '>'
    if (p.parseLess() || failed(parseDTypeParamValue(p, resultDType)) ||
        p.parseGreater())
      return {};
    return SIMDType::get(1, resultDType);
  });
}

//===----------------------------------------------------------------------===//
// ParamType
//===----------------------------------------------------------------------===//

Type ParamType::get(TypedAttr param) {
  // If the parameter is already resolved to a constant, fold this to the
  // indicated type. ParamType does not propagate the typeValue.
  //
  // NOTE: Strictly speaking, the folding below is only lossless when types
  // are consistent between the constant type value and the meta type, that is,
  // `constant.getMlirType().getMetaType() >= param.getType()`.  These do not
  // need to be the same, because we're very happy to strip off a Trait (like
  // Copyable) from a struct type (like Int) and we fold upcasts as well.
  //
  // FIXME: we should probably add some verification rules to verify the
  // property above since we are implicitly relying on the assumption here.  At
  // the KGEN level we can't verify this, but we could consider adding "is super
  // type of" style methods to ParameterTypeInterface.
  if (auto constant = dyn_cast<TypeParamAttr>(param))
    return constant.getMlirType();

  // If this is an trait upcast, we can look through it because we don't
  // propagate the typeValue, only the mlirType.
  //
  // NOTE: we can not look through trait downcast because the sole purpose of
  // downcast is to preserve both type value and the target type at the same
  // time.
  if (auto upcast = sugarDynCast<UpcastAttr>(param))
    return get(upcast.getInputTypeValue());

  // An extension is physically transparent: it augments the trait view but its
  // underlying physical type is its anchor.
  if (auto extension = sugarDynCast<ExtensionAttr>(param))
    return get(extension.getAnchor());

  // Otherwise, form the ParamType like normal.
  return Base::get(param.getContext(), param);
}

ParamType ParamType::getFromBytecode(TypedAttr param) {
  return Base::get(param.getContext(), param);
}

mlir::OpAsmAliasResult ParamType::getAlias(raw_ostream &os) const {
  // If this is a ParamType wrapping a simple SugarAttr then use an alias based
  // on the alias name to significantly compact the type reference.
  if (auto sugar = dyn_cast<SugarAttr>(getParam())) {
    if (sugar.getKind() == SugarKind::Alias)
      if (auto dre = dyn_cast<ParamDeclRefAttr>(sugar.getSugared())) {
        auto name = dre.getName().strref();
        // Remove mangling.
        name = name.take_front(name.find('`'));
        if (!name.empty()) {
          os << "alias_" << name;
          return mlir::OpAsmAliasResult::OverridableAlias;
        }
      }
  }
  return mlir::OpAsmAliasResult::NoAlias;
}

//===----------------------------------------------------------------------===//
// ModuleType
//===----------------------------------------------------------------------===//

OptionalParseResult ParamType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return {}; // No custom parsing.
}

LogicalResult ParamType::printValue(AsmPrinter &p, TypedAttr value) const {
  return failure(); // No custom printing.
}

//===----------------------------------------------------------------------===//
// TypeValueType
//===----------------------------------------------------------------------===//

Type TypeValueType::get(TypedAttr typeValue) {
  // Simply forward the value domain type from a TypeParamAttr.
  if (auto constant = dyn_cast<TypeParamAttr>(typeValue))
    return constant.getTypeValue();

  // Otherwise, form the TypeValueType like normal.
  return Base::get(typeValue.getContext(), typeValue);
}

TypeValueType TypeValueType::getFromBytecode(TypedAttr typeValue) {
  return Base::get(typeValue.getContext(), typeValue);
}

ArrayRef<TraitSymbolAttr> TypeValueType::getTraitSymbols() const {
  if (auto traitRef = sugarDynCast<TraitInstanceRefAttr>(getTypeValue()))
    return traitRef.getSymbols();
  return {};
}

//===----------------------------------------------------------------------===//
// NonStructTypeType
//===----------------------------------------------------------------------===//

OptionalParseResult NonStructTypeType::parseValue(AsmParser &p,
                                                  TypedAttr &value) const {
  return parseSugaredTypeValue(p, value, *this, parseOptionalKGENType);
}

LogicalResult NonStructTypeType::printValue(AsmPrinter &p,
                                            TypedAttr value) const {
  void (*typePrinter)(AsmPrinter &, Type) = &printKGENType; // Select overload.
  return printSugaredTypeValue(p, value, typePrinter);
}

OptionalParseResult TypeType::parseValue(AsmParser &p, TypedAttr &value) const {
  return parseSugaredTypeValue(p, value, *this, parseOptionalKGENType);
}

LogicalResult TypeType::printValue(AsmPrinter &p, TypedAttr value) const {
  void (*typePrinter)(AsmPrinter &, Type) = &printKGENType; // Select overload.
  return printSugaredTypeValue(p, value, typePrinter);
}

std::optional<int64_t> TypeType::getTypeSize(TargetInfoAttr target) const {
  // TODO: Types don't have a runtime representation yet! But one can imagine it
  // would contain a type ID, and a pointer to the witness table.

  // We use this size to allocate in the interpreter memory space
  // to store an opaque pointer which we always assume is of 64bit,
  // so we should allocate space for 64bit integers instead of
  // target specific pointer size which can be 32bit.
  return target.getDefaultPointerSize() * 2;
}

std::optional<int64_t> TypeType::getTypeAlign(TargetInfoAttr target) const {
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

ErrorOrSuccess TypeType::writeTo(TypedAttr value, int64_t addr,
                                 InterpreterState &state) const {
  return writeSymbolicAttribute(*this, value, addr, state);
}

ErrorOr<TypedAttr> TypeType::readFrom(int64_t addr,
                                      InterpreterState &state) const {
  return readSymbolicAttribute(*this, addr, state);
}

//===----------------------------------------------------------------------===//
// GeneratorType
//===----------------------------------------------------------------------===//

GeneratorType GeneratorType::getWithBody(Type newBody) {
  return GeneratorType::get(getInputParamTypes(), newBody, getParamListAttrs());
}

StringAttr GeneratorType::getParamName(size_t idx) {
  return getParamListAttrs().getName(idx);
}

ArrayRef<ConstraintAttr> GeneratorType::getBodyConstraints() {
  PogListAttr paramListAttrs = getParamListAttrs();
  if (!paramListAttrs)
    return {};
  return paramListAttrs.getBodyConstraints();
}

GeneratorType GeneratorType::getWithoutBodyConstraints() {
  PogListAttr paramListAttrs = getParamListAttrs();
  if (!paramListAttrs || paramListAttrs.getBodyConstraints().empty())
    return *this;
  PogListAttr stripped = PogListAttr::get(
      getContext(), paramListAttrs.getPogs(),
      /*bodyConstraints=*/{}, paramListAttrs.getOrigVariadicConvention());
  return GeneratorType::get(getInputParamTypes(), getBody(), stripped);
}

Type GeneratorType::parse(AsmParser &p) {
  GeneratorType generator;
  if (p.parseLess() || parseGenerator(p, generator) || p.parseGreater())
    return {};
  return generator;
}

void GeneratorType::print(AsmPrinter &p) const {
  p << '<';
  printGenerator(p, *this);
  p << '>';
}

OptionalParseResult GeneratorType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  if (auto sigGen = dyn_cast<FuncTypeGeneratorType>(*this)) {
    // Parse a keyword or string as an MLIR operation attribute.
    std::string opName;
    llvm::SMLoc loc = p.getCurrentLocation();
    if (succeeded(p.parseOptionalString(&opName))) {
      NamedAttrList attrs;
      if (failed(p.parseOptionalAttrDict(attrs)))
        return failure();
      value =
          MLIROpAttr::getChecked([&] { return p.emitError(loc); },
                                 StringAttr::get(p.getContext(), opName),
                                 attrs.getDictionary(p.getContext()), sigGen);
      return mlir::success(!!value);
    }

    Attribute attr;
    OptionalParseResult result = p.parseOptionalAttribute(attr, sigGen);
    if (!result.has_value())
      return std::nullopt;
    if (failed(*result))
      return failure();

    // Parse a symbol reference as a signature type attribute.
    if (auto symbol = dyn_cast<SymbolRefAttr>(attr)) {
      // Parse any trailing parameter bindings.
      ParameterExprArrayAttr paramValues;
      if (parseParameterValues(p, paramValues))
        return failure();
      value = SymbolConstantAttr::get(symbol, sigGen, paramValues);
    } else if (auto closureSymbol = dyn_cast<ClosureSymbolAttr>(attr)) {
      value = ClosureSymbolAttr::get(
          closureSymbol.getContext(), closureSymbol.getParentSymbol(),
          closureSymbol.getNestedFuncName(), closureSymbol.getMethod(),
          closureSymbol.getParamValues(), sigGen);
    } else {
      value = llvm::cast<TypedAttr>(attr);
    }
    return mlir::success();
  }

  return {};
}

LogicalResult GeneratorType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (::isa<FuncTypeGeneratorType>(*this)) {
    if (auto mlirOp = dyn_cast<MLIROpAttr>(value)) {
      p << mlirOp.getName();
      if (!mlirOp.getAttrs().empty())
        p << mlirOp.getAttrs();
      return success();
    }

    auto symbolCst = dyn_cast<SymbolConstantAttr>(value);
    if (!symbolCst)
      return failure();
    p << symbolCst.getSymbol();
    printParameterValues(p, symbolCst.getParamValues());
    return success();
  }

  return failure();
}

std::optional<int64_t> GeneratorType::getTypeSize(TargetInfoAttr target) const {
  // Temporary back-compat: delegate to FuncType.
  if (auto sig = dyn_cast<FuncType>(getBody()))
    return sig.getTypeSize(target);
  return std::nullopt;
}

std::optional<int64_t>
GeneratorType::getTypeAlign(TargetInfoAttr target) const {
  // Temporary back-compat: delegate to FuncType.
  if (auto sig = dyn_cast<FuncType>(getBody()))
    return sig.getTypeAlign(target);
  return std::nullopt;
}

ErrorOrSuccess GeneratorType::writeTo(TypedAttr value, int64_t addr,
                                      InterpreterState &state) const {
  // Temporary back-compat: delegate to FuncType.
  if (auto sig = dyn_cast_if_present<FuncType>(getBody()))
    return sig.writeTo(value, addr, state);

  return Error("generator not a writeable type, got " +
               mlir::debugString(value) + " instead");
}

ErrorOr<TypedAttr> GeneratorType::readFrom(int64_t addr,
                                           InterpreterState &state) const {
  // Temporary back-compat: delegate to FuncType.
  if (auto sig = dyn_cast<FuncType>(getBody()))
    return sig.readFrom(addr, state);
  return Error("Generator not a readable type");
}

GeneratorType GeneratorType::getSpecializedGenerator(
    PartiallySpecializedInputParams &specialization,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  PogListAttr genMetadata = getParamListAttrs();
  if (genMetadata) {
    PogListAttr specialized = genMetadata.getSpecializedMetadata(
        specialization.evaluator, specialization.boundParams, emitErrorFn,
        specialization.dischargedBodyConstraints);
    if (!specialized)
      return {};
    genMetadata = specialized;
  }

  return GeneratorType::get(specialization.unboundParamTypes,
                            specialization.evaluator.getReboundType(getBody()),
                            genMetadata);
}

GeneratorType GeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  VerboseCompilerTimeTraceScope traceScope(
      "GeneratorType::getSpecializedGenerator");

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  if (paramBindings.empty())
    return *this;

  std::optional<PartiallySpecializedInputParams> specializationOpt =
      PartiallySpecializedInputParams::from(getInputParamTypes(), paramBindings,
                                            evaluationContext, emitErrorFn);
  if (!specializationOpt)
    return {};

  return getSpecializedGenerator(*specializationOpt, emitErrorFn);
}

GeneratorType GeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext, Location location) {
  return getSpecializedGenerator(
      paramBindings, evaluationContext,
      [&]() -> InFlightDiagnostic { return emitError(location); });
}

//===----------------------------------------------------------------------===//
// FuncGeneratorTypeBuilderType
//===----------------------------------------------------------------------===//

Type FuncGeneratorTypeBuilderType::get(MLIRContext *ctx, TypedAttr paramDecls,
                                       TypedAttr argTypes, TypedAttr resultType,
                                       TypedAttr metadata,
                                       TypedAttr paramListAttrs,
                                       TypedAttr argListAttrs) {
  auto cstParamDecls = dyn_cast<ParamListAttr>(paramDecls);
  auto cstArgTypes = dyn_cast<ParamListAttr>(argTypes);
  auto cstMetadata = dyn_cast<FnMetadataAttr>(metadata);

  // Two optional pogs.
  auto cstParamListAttrs = dyn_cast_or_null<PogListAttr>(paramListAttrs);
  auto cstArgListAttrs = dyn_cast_or_null<PogListAttr>(argListAttrs);

  // If any of the components are not constants, skip.
  if (!cstParamDecls || !cstArgTypes || !cstMetadata ||
      (paramListAttrs && !cstParamListAttrs) || // has a non-constant param pog
      (argListAttrs && !cstArgListAttrs))       // has a non-constant arg pog
    return Base::get(ctx, paramDecls, argTypes, resultType, metadata,
                     paramListAttrs, argListAttrs);
  assert(llvm::all_of(cstParamDecls.getValues(), llvm::IsaPred<QuoteAttr>) &&
         "malformed fn gen builder param decls");

  // Unquote before we assemble a FnTypeGeneratorType.
  auto unquote = [](TypedAttr value) -> TypedAttr {
    return cast<QuoteAttr>(value).getQuotedParam();
  };

  SmallVector<Type> inputParamTypes = llvm::map_to_vector(
      cstParamDecls.getValues(),
      [&](TypedAttr attr) -> Type { return ParamType::get(unquote(attr)); });

  SmallVector<Type> inputArgTypes =
      llvm::map_to_vector(cstArgTypes.getValues(), [&](TypedAttr attr) -> Type {
        return ParamType::get(unquote(attr));
      });

  // Fold to the generator type this builder describes.
  return KGEN::FuncTypeGeneratorType::get(
      inputParamTypes,
      FuncType::get(FunctionType::get(ctx, inputArgTypes,
                                      ParamType::get(unquote(resultType))),
                    cstMetadata.getArgConventions(), cstMetadata.getFnEffects(),
                    cstMetadata.getMetadata(), cstArgListAttrs),
      cstParamListAttrs);
}

Type FuncGeneratorTypeBuilderType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *ctx,
    TypedAttr paramDecls, TypedAttr argTypes, TypedAttr resultType,
    TypedAttr metadata, TypedAttr paramListAttrs, TypedAttr argListAttrs) {
  if (failed(verify(emitError, paramDecls, argTypes, resultType, metadata,
                    paramListAttrs, argListAttrs)))
    return {};
  return get(ctx, paramDecls, argTypes, resultType, metadata, paramListAttrs,
             argListAttrs);
}

LogicalResult FuncGeneratorTypeBuilderType::verify(
    function_ref<InFlightDiagnostic()> emitError, TypedAttr paramDecls,
    TypedAttr argTypes, TypedAttr resultType, TypedAttr metadata,
    TypedAttr paramListAttrs, TypedAttr argListAttrs) {

  // NOTE:we can not easily verify that this is a list of type values (since
  // type expressions might in different forms between lit/kgen)
  if (!isa<ParamListType>(argTypes.getType()))
    return emitError()
           << " expect to have !kgen.param_list type for argument type list";

  if (!isa<ParamListType>(paramDecls.getType()))
    return emitError()
           << " expect to have !kgen.param_list type for parameter decl list";

  if (!isa<NonStructTypeType>(metadata.getType()))
    return emitError()
           << "function metadata should have !kgen.non_struct_type type, not "
           << metadata.getType();

  // The pog list components are typed `!kgen.non_struct_type` parameter
  // expressions: a concrete `#kgen.pog_list` or a symbolic reference.
  if (paramListAttrs && !isa<NonStructTypeType>(paramListAttrs.getType()))
    return emitError()
           << "parameter pog list should have !kgen.non_struct_type type, not "
           << paramListAttrs.getType();

  if (argListAttrs && !isa<NonStructTypeType>(argListAttrs.getType()))
    return emitError()
           << "argument pog list should have !kgen.non_struct_type type, not "
           << argListAttrs.getType();

  return success();
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

ArrayRef<Type> FuncType::getArguments() const {
  return getValues().getInputs();
}
ArrayRef<Type> FuncType::getResults() const { return getValues().getResults(); }

ArrayRef<ArgConvention> FuncType::getArgConventions() const {
  return getMetadataAttr().getArgConventions();
}
FnEffects FuncType::getFnEffects() const {
  return getMetadataAttr().getFnEffects();
}
FnMetadataAttrInterface FuncType::getMetadata() const {
  return getMetadataAttr().getMetadata();
}

bool FuncType::hasMemoryOnlyResult() {
  ArrayRef<ArgConvention> conventions = getArgConventions();
  return !conventions.empty() &&
         conventions.back() == ArgConvention::ByRefResult;
}

FuncType FuncType::getWithFnEffects(FnEffects effects) {
  return FuncType::get(getContext(), getValues(),
                       getMetadataAttr().getWithFnEffects(effects),
                       getArgListAttrs());
}
FuncType FuncType::getWithValuesReplaced(FunctionType fnType) {
  return FuncType::get(fnType, getArgConventions(), getFnEffects(),
                       getMetadata(), getArgListAttrs());
}

FuncType FuncType::getWithMetadata(FnMetadataAttrInterface metadata,
                                   PogListAttr argListAttrs) {
  if (!argListAttrs)
    argListAttrs = getArgListAttrs();
  return FuncType::get(getContext(), getValues(),
                       getMetadataAttr().getWithMetadata(metadata),
                       argListAttrs);
}

FuncType FuncType::get(FunctionType values, FnMetadataAttrInterface metadata,
                       PogListAttr argListAttrs) {
  return FuncType::get(values, {}, {}, metadata, argListAttrs);
}

size_t FuncType::getNumAsyncReturnSlots() {
  return isAsync() ? (hasMemoryOnlyResult() + isThrows()) : 0;
}

StringAttr FuncType::getArgName(size_t idx) {
  return getArgListAttrs().getName(idx);
}

bool FuncType::isAnyVarArg(size_t index) {
  return getArgListAttrs().isAnyVarArg(index);
}

bool FuncType::isPosVarArg(size_t index) {
  return getArgListAttrs().isPosVarArg(index);
}

/// For a PosVarArg, return the declared ArgConvention of the elements. For
/// example: def x(mut *args: Int) is declared 'mut'.
ArgConvention FuncType::getVariadicConvention(size_t index) {
  PogListAttr pogs = getArgListAttrs();
  if (pogs.getVariadicKind(index) == VariadicKind::PosVarArg ||
      pogs.getVariadicKind(index) == VariadicKind::PackVarArg)
    return pogs.getOrigVariadicConvention();
  return ArgConvention::ByRefError;
}

bool FuncType::isKwVarArg(size_t index) {
  return getArgListAttrs().isKwVarArg(index);
}

bool FuncType::isPack(size_t index) { return getArgListAttrs().isPack(index); }

std::optional<size_t> FuncType::findPackVarArgIndex() {
  size_t numUserArgs = getNumArguments() - hasMemoryOnlyResult() - isThrows();
  if (numUserArgs == 0)
    return std::nullopt;
  size_t lastUserArgIndex = numUserArgs - 1;
  if (isKwVarArg(lastUserArgIndex)) {
    if (lastUserArgIndex == 0)
      return std::nullopt;
    --lastUserArgIndex;
  }
  if (isPack(lastUserArgIndex))
    return std::make_optional(lastUserArgIndex);
  return std::nullopt;
}

bool FuncType::hasKwVarArgs() { return getArgListAttrs().hasKwVarArg(); }

std::optional<int64_t> FuncType::getTypeSize(TargetInfoAttr target) const {
  // Non-capturing closures are function pointers. Capturing closures contain
  // a function pointer and a capture state pointer.
  return (isCapturing() ? 2 : 1) * target.getDataLayout().getPointerSize();
}

std::optional<int64_t> FuncType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

ErrorOrSuccess FuncType::writeTo(TypedAttr value, int64_t addr,
                                 InterpreterState &state) const {
  // The index is written to the slot.
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<void *> mem =
      state.getWritableMemory(addr, ptrSize, RegionMark::Symbol);
  if (mem.isError())
    return mem.takeError();

  // Store the actual symbol in symbolic memory
  uint64_t index = state.addSymbolToSymbolTable(value);

  // Store the index of the symbol in the pointer slot.
  llvm::StoreIntToMemory(APInt(ptrSize * 8, index), (uint8_t *)*mem, ptrSize);
  return success();
}

ErrorOr<TypedAttr> FuncType::readFrom(int64_t addr,
                                      InterpreterState &state) const {
  // The index is written to the slot.
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<const void *> mem = state.getReadableMemory(addr, ptrSize);
  if (mem)
    return mem.takeError();

  APInt value(ptrSize * 8, 0);
  llvm::LoadIntFromMemory(value, (const uint8_t *)*mem, ptrSize);
  ErrorOr<TypedAttr> symbol = state.getSymbol(value.getZExtValue());
  if (!symbol.isError()) {
    if (auto generatorType =
            dyn_cast<M::KGEN::FuncTypeGeneratorType>(symbol->getType())) {
      // structs that contain 'recursive' function pointers are not true
      // recursion but require one level of typing to preserve symbol storage in
      // the interpreter. Handle the resulting type mismatches with a bit cast.
      if (generatorType.getBody() != *this) {
        assert(isa<SymbolConstantAttr>(*symbol) &&
               "func slot must hold a symbol constant");
        auto symbolCst = cast<SymbolConstantAttr>(*symbol);
        auto shallowGen =
            cast<FuncTypeGeneratorType>(generatorType.getWithBody(*this));
        symbol =
            cast<TypedAttr>(FuncPtrBitcastAttr::get(symbolCst, shallowGen));
      }
    }
  }
  return symbol;
}

Type FuncType::parse(AsmParser &parser) {
  FuncType signature;
  if (parser.parseLess() || parseFuncType(parser, signature) ||
      parser.parseGreater())
    return {};
  return signature;
}

void FuncType::print(AsmPrinter &printer) const {
  printer << '<';
  printFuncType(printer, *this);
  printer << '>';
}

FuncType FuncType::get(MLIRContext *context, TypeRange inputs,
                       TypeRange results) {
  return get(FunctionType::get(context, inputs, results));
}

FuncType FuncType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, TypeRange inputs,
                              TypeRange results) {
  auto result = get(context, inputs, results);
  if (failed(verify(emitError, result.getValues(), result.getMetadataAttr(),
                    result.getArgListAttrs())))
    return {};
  return result;
}

LogicalResult FuncType::verify(function_ref<InFlightDiagnostic()> emitError,
                               FunctionType values, FnMetadataAttr metadataAttr,
                               PogListAttr argListAttrs) {
  if (!metadataAttr)
    return emitError() << "func type requires a metadata attribute";

  ArrayRef<ArgConvention> argConventions = metadataAttr.getArgConventions();
  FnEffects effects = metadataAttr.getFnEffects();
  FnMetadataAttrInterface metadata = metadataAttr.getMetadata();

  // Check we have the right number of conventions.
  if (argConventions.size() != values.getInputs().size())
    return emitError() << "incorrect # of input conventions specified";

  // If `argListAttrs` is populated, its size must match the number of inputs.
  // An empty list is the "no source-level metadata" case and is allowed.
  if (size_t numPogs = argListAttrs.getPogs().size();
      numPogs != 0 && numPogs != values.getNumInputs()) {
    return emitError()
           << "number of arguments does not match number of argument names: "
           << values.getNumInputs() << " != " << numPogs;
  }

  // If the FuncType has metadata, defer to it for further verification.
  // Otherwise, run the standard KGEN FuncType verification.
  if (metadata)
    return metadata.verifyFuncType(emitError, values, argConventions, effects);

  // Verify input convention and argument types.
  int64_t byRefResultIdx = -1;
  int64_t byRefErrorIdx = -1;
  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), argConventions)) {
    if (conv == ArgConvention::ByRefResult) {
      if (byRefResultIdx >= 0)
        return emitError() << "func type cannot have more than one argument "
                              "with 'byref_result'";
      byRefResultIdx = i;
    }
    if (conv == ArgConvention::ByRefError) {
      if (byRefErrorIdx >= 0)
        return emitError() << "func type cannot have more than one argument "
                              "with 'byref_error'";
      if (!effects.isThrows())
        return emitError()
               << "func type with 'byref_error' argument must be 'throws'";
      byRefErrorIdx = i;
    }

    Type type = argType;
    // Verify variadics.
    if (auto variadic = dyn_cast<ParamListType>(type))
      type = variadic.getElementType();
    // Verify argument conventions.  Before lit lowering, they need to be
    // !lit.ref type, after lowering, they should have !kgen.pointer type.
    if (hasAddress(conv)) {
      if (::isa<PointerType>(type))
        continue;
      // TODO: During LowerLIT, we strip off the metadata, but later we lower
      // references to pointers.  This means that LowerLIT needs a !kgen.func
      // (without LIT attribute) with references.  Accept !lit.ref until we can
      // sort this out.
      if (type.getDialect().getNamespace() == "lit")
        continue;

      return emitError()
             << "argument #" << i << " with convention '" << stringifyEnum(conv)
             << "' in func type should be a `!kgen.pointer` but got: " << type;
    }
  }

  bool hasByRefResult = byRefResultIdx >= 0;
  bool hasByRefError = byRefErrorIdx >= 0;
  if (hasByRefResult && byRefResultIdx != values.getNumInputs() - 1)
    return emitError() << "'byref_result' argument must be the last argument";
  if (hasByRefError &&
      byRefErrorIdx != values.getNumInputs() - 1 - hasByRefResult)
    return emitError() << "'byref_error' argument must be the last "
                          "argument before any `byref_result` argument";

  return success();
}

//===----------------------------------------------------------------------===//
// FuncLiteralType
//===----------------------------------------------------------------------===//

OptionalParseResult FuncLiteralType::parseValue(AsmParser &p,
                                                TypedAttr &value) const {
  // TODO: We can pretty print/parse a constant function literal.
  return {};
}

LogicalResult FuncLiteralType::printValue(AsmPrinter &p,
                                          TypedAttr value) const {
  return failure();
}

FuncSymbolAttr FuncLiteralType::getTargetLiteral() const {
  return sugarCast<FuncSymbolAttr>(getFuncLiteral());
}

LogicalResult
FuncLiteralType::verify(function_ref<InFlightDiagnostic()> emitError,
                        TypedAttr value) {
  // TODO: we can potentially allow non-constant expressions here too.
  if (!sugarIsaAndNonNull<FuncSymbolAttr>(value))
    return emitError() << "expected a FuncSymbolAttr attribute to construct a "
                          "FuncLiteralType";

  return success();
}

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

FuncTypeGeneratorType::FuncTypeGeneratorType(GeneratorType gen)
    : GeneratorType(gen) {
  assert((!gen || ::isa_and_nonnull<FuncType>(gen.getBody())) &&
         "expected FuncType as body");
}

FuncTypeGeneratorType
FuncTypeGeneratorType::get(ArrayRef<Type> inputParamTypes, FunctionType values,
                           ArrayRef<ArgConvention> argConvs, FnEffects effects,
                           Attribute fnMetadata, Attribute genMetadata,
                           Attribute argListAttrs) {
  auto sig = FuncType::get(values, argConvs, effects,
                           ::cast_or_null<FnMetadataAttrInterface>(fnMetadata),
                           argListAttrs);
  return FuncTypeGeneratorType(GeneratorType::get(
      inputParamTypes, sig, ::cast_or_null<PogListAttr>(genMetadata)));
}

FuncTypeGeneratorType FuncTypeGeneratorType::get(ArrayRef<Type> inputParamTypes,
                                                 FuncType sig,
                                                 Attribute genMetadata) {
  return FuncTypeGeneratorType(GeneratorType::get(
      inputParamTypes, sig, ::cast_or_null<PogListAttr>(genMetadata)));
}

FuncTypeGeneratorType FuncTypeGeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  return ::cast_or_null<FuncTypeGeneratorType>(
      GeneratorType::getSpecializedGenerator(paramBindings, evaluationContext,
                                             emitErrorFn));
}

FuncTypeGeneratorType FuncTypeGeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext, Location location) {
  return ::cast_or_null<FuncTypeGeneratorType>(
      GeneratorType::getSpecializedGenerator(paramBindings, evaluationContext,
                                             location));
}

FuncTypeGeneratorType FuncTypeGeneratorType::remapToFuncTypeGenerator(
    ArrayRef<ParamDeclAttr> inputParams, FunctionType functionType,
    ArrayRef<ArgConvention> argConventions, FnEffects effects,
    Attribute fnMetadata, Attribute genMetadata,
    function_ref<InFlightDiagnostic()> emitError, Attribute argListAttrs) {
  IndexRefRemapper remapper(inputParams, {});
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : inputParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));

  auto emitter = []() -> InFlightDiagnostic {
    llvm_unreachable("invalid func type generator");
  };

  if (!emitError)
    emitError = emitter;

  auto newSig = FuncType::getChecked(
      emitError, remapper.replace(functionType), argConventions, effects,
      fnMetadata ? remapper.replace(fnMetadata) : nullptr,
      argListAttrs ? remapper.replace(argListAttrs) : nullptr);
  if (!newSig)
    return nullptr;
  return GeneratorType::get(inputParamTypes, newSig,
                            genMetadata ? remapper.replace(genMetadata)
                                        : nullptr);
}

FuncType FuncTypeGeneratorType::getBody() {
  return ::cast<FuncType>(GeneratorType::getBody());
}

FuncType FuncTypeGeneratorType::getInstantiatedBody() {
  return ::cast<FuncType>(GeneratorType::getInstantiatedBody());
}

bool FuncTypeGeneratorType::classof(GeneratorType type) {
  return ::isa_and_nonnull<FuncType>(type.getBody());
}

bool FuncTypeGeneratorType::classof(Type type) {
  if (auto gen = dyn_cast<GeneratorType>(type))
    return classof(gen);
  return false;
}

//===----------------------------------------------------------------------===//
// FuncLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

FuncLiteralTypeGeneratorType::FuncLiteralTypeGeneratorType(GeneratorType gen)
    : GeneratorType(gen) {
  assert((!gen || ::isa_and_nonnull<FuncLiteralType>(gen.getBody())) &&
         "expected FuncLiteralType as body");
}

FuncLiteralTypeGeneratorType
FuncLiteralTypeGeneratorType::get(ArrayRef<Type> inputParamTypes,
                                  TypedAttr funcLiteral,
                                  Attribute genMetadata) {

  assert(isa_and_nonnull<FuncType>(funcLiteral.getType()) &&
         "expected a func literal with FuncType");
  auto funcLiteralType = FuncLiteralType::get(funcLiteral);
  return FuncLiteralTypeGeneratorType(
      GeneratorType::get(inputParamTypes, funcLiteralType, genMetadata));
}

FuncLiteralTypeGeneratorType
FuncLiteralTypeGeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  return ::cast_or_null<FuncLiteralTypeGeneratorType>(
      GeneratorType::getSpecializedGenerator(paramBindings, evaluationContext,
                                             emitErrorFn));
}

FuncLiteralTypeGeneratorType
FuncLiteralTypeGeneratorType::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext, Location location) {
  return ::cast_or_null<FuncLiteralTypeGeneratorType>(
      GeneratorType::getSpecializedGenerator(paramBindings, evaluationContext,
                                             location));
}

SymbolConstantAttr FuncLiteralTypeGeneratorType::getSymbolConstantAttr() {
  auto directSymbol = getBody().getTargetLiteral();
  auto sig = GeneratorType::get(getInputParamTypes(), directSymbol.getType(),
                                getParamListAttrs());

  // The parameter values might have dependent type, erase the ref too in the
  // create a evaluator to erase the dependencies too.
  ParameterEvaluator evaluator(ArrayRef<TypedAttr>{});
  for (auto unboundType : getInputParamTypes()) {
    // There could be a dependent input parameter type, erase the unknown ref.
    auto unbound = UnboundAttr::get(evaluator.replace(unboundType));
    evaluator.appendIndexBinding(unbound);
  }

  SmallVector<TypedAttr> paramValues;
  for (auto paramValue : directSymbol.getParamValues())
    paramValues.push_back(evaluator.replace(paramValue));

  return SymbolConstantAttr::get(directSymbol.getSymbol(),
                                 cast<FuncTypeGeneratorType>(sig), paramValues);
}

FuncLiteralType FuncLiteralTypeGeneratorType::getBody() {
  return ::cast<FuncLiteralType>(GeneratorType::getBody());
}

FuncLiteralType FuncLiteralTypeGeneratorType::getInstantiatedBody() {
  return ::cast<FuncLiteralType>(GeneratorType::getInstantiatedBody());
}

bool FuncLiteralTypeGeneratorType::classof(GeneratorType type) {
  return ::isa_and_nonnull<FuncLiteralType>(type.getBody());
}

bool FuncLiteralTypeGeneratorType::classof(Type type) {
  if (auto gen = dyn_cast<GeneratorType>(type))
    return classof(gen);
  return false;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// Custom parser for optional addressSpace and isNonNull parameters.
// Supports: !kgen.pointer<T>, !kgen.pointer<T, addrSpace>,
//           !kgen.pointer<T, addrSpace, nonnull>, !kgen.pointer<T, nonnull>
static ParseResult parsePointerOptionals(AsmParser &p, TypedAttr &addressSpace,
                                         bool &isNonNull) {
  // Set defaults
  auto builder = p.getBuilder();
  addressSpace = builder.getIndexAttr(0);
  isNonNull = false;

  while (succeeded(p.parseOptionalComma())) {
    // Try nonnull keyword first before parsing as address space
    if (succeeded(p.parseOptionalKeyword("nonnull"))) {
      isNonNull = true;
      continue;
    }
    // AddressSpace (try this after keyword parsing)
    if (succeeded(parseIndexParamValue(p, addressSpace))) {
      continue;
    }
    // TODO:
    // dereferenceable<N>
    // dereferenceable_or_null<N>
  }

  return success();
}

static void printPointerOptionals(AsmPrinter &p, TypedAttr addressSpace,
                                  bool isNonNull) {
  // Get default values for comparison
  auto defaultAddrSpace =
      IntegerAttr::get(IndexType::get(addressSpace.getContext()), 0);

  if (addressSpace != defaultAddrSpace) {
    p << ", ";
    printIndexParamValue(p, addressSpace);
  }

  if (isNonNull) {
    p << ", nonnull";
  }
}

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, TypedAttr addressSpace,
                                  bool isNonNull) {
  if (!addressSpace.getType().isIndex()) {
    return emitError() << "pointer address space parameter `" << addressSpace
                       << "` must be an index type";
  }
  // isNonNull is always valid (bool), no verification needed
  return success();
}

PointerType PointerType::get(Type elementType, unsigned addressSpace,
                             bool isNonNull) {
  Builder b(elementType.getContext());
  return get(elementType.getContext(), elementType,
             b.getIndexAttr(addressSpace), isNonNull);
}

PointerType
PointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        Type elementType, unsigned addressSpace,
                        bool isNonNull) {
  Builder b(elementType.getContext());
  return getChecked(emitError, elementType.getContext(), elementType,
                    b.getIndexAttr(addressSpace), isNonNull);
}

PointerType PointerType::get(Type elementType, TypedAttr addressSpace,
                             bool isNonNull) {
  return get(elementType.getContext(), elementType, addressSpace, isNonNull);
}

PointerType
PointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        Type elementType, TypedAttr addressSpace,
                        bool isNonNull) {
  return getChecked(emitError, elementType.getContext(), elementType,
                    addressSpace, isNonNull);
}

std::optional<int64_t> PointerType::getTypeSize(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerSize(getAddrSpaceOrZero());
}

std::optional<int64_t> PointerType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign(getAddrSpaceOrZero());
}

ErrorOrSuccess PointerType::writeTo(TypedAttr value, int64_t addr,
                                    InterpreterState &state) const {
  int64_t size = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem = state.getWritableMemory(addr, size, RegionMark::Pointer,
                                                getAddrSpaceOrZero());
  if (mem.isError())
    return mem.takeError();

  // The pointer size of the target is variable.
  if (auto pv = dyn_cast_if_present<PointerAttr>(value)) {
    APInt intVal(size * CHAR_BIT, pv.getAddr());
    llvm::StoreIntToMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
    return success();
  }

  return Error("pointer not a writeable type, got " + mlir::debugString(value) +
               " instead");
}

ErrorOr<TypedAttr> PointerType::readFrom(int64_t addr,
                                         InterpreterState &state) const {
  int64_t size = *getTypeSize(state.getTarget());
  ErrorOr<const void *> mem = state.getReadableMemory(addr, size);
  if (mem.isError())
    return mem.takeError();
  APInt intVal(size * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(intVal, (const uint8_t *)*mem, size);
  return PointerAttr::get(intVal.getLimitedValue(), *this);
}

OptionalParseResult PointerType::parseValue(AsmParser &p,
                                            TypedAttr &value) const {
  int64_t addr = 0;
  // Parse an integer as a raw pointer attribute.
  if (OptionalParseResult result = p.parseOptionalInteger(addr);
      result.has_value()) {
    if (failed(*result))
      return failure();
    value = PointerAttr::get(addr, *this);
    return mlir::success();
  }

  // Parse a `store_to_mem` directive.
  if (succeeded(p.parseOptionalKeyword("store_to_mem"))) {
    TypedAttr memValue;
    if (p.parseLParen() || parseParamValue(p, memValue, getElementType()) ||
        p.parseRParen())
      return failure();
    value = StoreToMemAttr::get(memValue, *this);
    return mlir::success();
  }

  return {};
}

LogicalResult PointerType::printValue(AsmPrinter &p, TypedAttr value) const {
  // Print a raw pointer attribute as an integer.
  if (auto ptrAttr = dyn_cast<PointerAttr>(value)) {
    p << ptrAttr.getAddr();
    return success();
  }

  // Print a `store_to_mem` directive.
  if (auto memAttr = dyn_cast<StoreToMemAttr>(value)) {
    p << "store_to_mem(";
    printParamValue(p, memAttr.getValue());
    p << ')';
    return success();
  }

  return failure();
}

std::optional<unsigned> PointerType::getAddrSpace() const {
  if (TypedAttr addrSpaceAttr = getAddressSpace()) {
    if (auto addrSpaceIntAttr = dyn_cast<IntegerAttr>(addrSpaceAttr))
      return addrSpaceIntAttr.getInt();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

std::optional<int64_t> DTypeType::getTypeSize(TargetInfoAttr target) const {
  return sizeof(uint8_t);
}

std::optional<int64_t> DTypeType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

ErrorOrSuccess DTypeType::writeTo(TypedAttr value, int64_t addr,
                                  InterpreterState &state) const {
  // DType is one byte.
  ErrorOr<void *> mem = state.getWritableMemory(addr, 1);
  if (mem.isError())
    return mem.takeError();
  if (auto dv = dyn_cast_if_present<DTypeConstantAttr>(value)) {
    *(uint8_t *)*mem = dv.getDType().getValue();
    return success();
  }

  return Error("dtype not a writeable type, got " + mlir::debugString(value) +
               " instead");
}

ErrorOr<TypedAttr> DTypeType::readFrom(int64_t addr,
                                       InterpreterState &state) const {
  ErrorOr<const void *> mem = state.getReadableMemory(addr, 1);
  if (mem.isError())
    return mem.takeError();
  return DTypeConstantAttr::get(getContext(),
                                KGENDType(*(const uint8_t *)*mem));
}

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

std::optional<int64_t>
KGEN::NoneType::getTypeSize(TargetInfoAttr target) const {
  return 0;
}

std::optional<int64_t>
KGEN::NoneType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

ErrorOrSuccess KGEN::NoneType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {
  return success();
}

ErrorOr<TypedAttr> KGEN::NoneType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  return KGEN::NoneAttr::get(state.getContext());
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

// A StringType is implemented as struct {char *address; size_t size;}.
// An index type as same alignment and size of a pointer type.
std::optional<int64_t>
KGEN::StringType::getTypeSize(TargetInfoAttr target) const {
  // We use this size to allocate in the interpreter memory space
  // to store the pointer points to the string and its size.
  return 2 * llvm::alignTo(target.getDataLayout().getPointerSize(),
                           target.getDataLayout().getPointerABIAlign());
}

std::optional<int64_t>
KGEN::StringType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

/// Helper to write a data pointer plus size type, both of which are pointer
/// width, and the pointer comes first.
static ErrorOrSuccess writePointerAndSize(int64_t writeAddr, int64_t ptr,
                                          int64_t size,
                                          InterpreterState &state) {
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<void *> mem = state.getWritableMemory(writeAddr, ptrSize * 2);
  if (mem.isError())
    return mem.takeError();
  // Store the pointer address, and then advance a pointer width and store the
  // size.
  llvm::StoreIntToMemory(APInt(ptrSize * 8, ptr), (uint8_t *)*mem, ptrSize);
  llvm::StoreIntToMemory(APInt(ptrSize * 8, size), (uint8_t *)*mem + ptrSize,
                         ptrSize);
  return success();
}

/// Helper to read a data pointer and size type, both of which are pointer
/// width, and the pointer comes first.
static ErrorOr<std::pair<int64_t, int64_t>>
readPointerAndSize(int64_t readAddr, InterpreterState &state) {
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<const void *> mem = state.getReadableMemory(readAddr, ptrSize * 2);
  if (mem.isError())
    return mem.takeError();
  APInt ptrVal(ptrSize * 8, 0);
  APInt sizeVal(ptrSize * 8, 0);
  llvm::LoadIntFromMemory(ptrVal, (const uint8_t *)*mem, ptrSize);
  llvm::LoadIntFromMemory(sizeVal, (const uint8_t *)*mem + ptrSize, ptrSize);
  return std::make_pair(ptrVal.getLimitedValue(), sizeVal.getLimitedValue());
}

ErrorOrSuccess StringType::writeTo(TypedAttr value, int64_t addr,
                                   InterpreterState &state) const {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  if (auto strAttr = dyn_cast_if_present<StringAttr>(value)) {
    StringRef str(strAttr.data(), strAttr.size() + 1);
    if (strAttr.getValue().empty())
      str = "\0";
    MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
    ErrorOr<int64_t> strAddr = state.mapConstGlobalMemory(hdl);
    if (strAddr.isError())
      return strAddr.takeError();

    // Store a pointer and a size.
    return writePointerAndSize(addr, *strAddr, strAttr.size(), state);
  }

  return Error("string not a writeable type, got " + mlir::debugString(value) +
               " instead");
}

ErrorOr<TypedAttr> StringType::readFrom(int64_t addr,
                                        InterpreterState &state) const {
  // Load a pointer and size.
  ErrorOr<std::pair<int64_t, int64_t>> ptrSize =
      readPointerAndSize(addr, state);
  if (ptrSize)
    return ptrSize.takeError();
  auto [strAddr, strSize] = *ptrSize;

  // Read back the string.
  ErrorOr<const void *> strMem = state.getReadableMemory(strAddr, strSize);
  if (strMem.isError())
    return strMem.takeError();

  return StringAttr::get(StringRef((const char *)*strMem, strSize), *this);
}

//===----------------------------------------------------------------------===//
// ParamListType
//===----------------------------------------------------------------------===//

/// A variadic type is like an `llvm::ArrayRef`: a pointer to the start of the
/// contiguous sequence, and the size of that sequence. So, its size would be
/// the size of a pointer, plus the size of the size type (which has the same
/// size and alignment as a pointer type).
std::optional<int64_t> ParamListType::getTypeSize(TargetInfoAttr target) const {
  // We use this size to allocate in the interpreter memory space
  // to store the pointer points to the start of the contiguous sequence
  // and its size.
  return 2 * llvm::alignTo(target.getDataLayout().getPointerSize(),
                           target.getDataLayout().getPointerABIAlign());
}

/// The alignment of the variadic type is that its pointer and size.
std::optional<int64_t>
ParamListType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

ErrorOrSuccess ParamListType::writeTo(TypedAttr value, int64_t addr,
                                      InterpreterState &state) const {
  // A variadic is a pointer and a size, where the pointer refers to
  // stack-allocated memory.
  auto vv = dyn_cast_if_present<ParamListAttr>(value);
  if (!vv) {
    return Error("variadic not a writeable type, got " +
                 mlir::debugString(value) + " instead");
  }

  ArrayRef<TypedAttr> values = vv.getValues();
  TargetInfoAttr target = state.getTarget();

  // Query the size and alignment of the element type.
  Type elemType = getElementType();
  std::optional<int64_t> typeAlign =
      DataLayoutInterface::getTypeABIAlign(target, elemType);
  std::optional<int64_t> allocSize =
      DataLayoutInterface::getTypeAllocSize(target, elemType);
  if (!typeAlign || !allocSize)
    return Error("could not query element type size or alignment");

  // Allocate stack memory for the elements.
  ErrorOr<int64_t> valuesAddr =
      state.allocatePersistentMemory(*allocSize * values.size(), *typeAlign);
  if (valuesAddr.isError())
    return valuesAddr.takeError();
  int64_t baseAddr = *valuesAddr;

  // Now write all the elements to the stack memory.
  for (auto [i, value] : llvm::enumerate(values)) {
    if (ErrorOrSuccess err =
            state.writeAttributeToMemory(baseAddr + i * *allocSize, value))
      return err.takeError();
  }

  // And now write the pointer and size.
  return writePointerAndSize(addr, baseAddr, values.size(), state);
}

ErrorOr<TypedAttr> ParamListType::readFrom(int64_t addr,
                                           InterpreterState &state) const {
  // Read the pointer and size.
  ErrorOr<std::pair<int64_t, int64_t>> ptrSize =
      readPointerAndSize(addr, state);
  if (ptrSize)
    return ptrSize.takeError();
  auto [baseAddr, numElems] = *ptrSize;

  // Query the size and alignment of the element type.
  TargetInfoAttr target = state.getTarget();
  Type elemType = getElementType();
  std::optional<int64_t> allocSize =
      DataLayoutInterface::getTypeAllocSize(target, elemType);
  if (!allocSize)
    return Error("could not query element type size or alignment");

  // Now read the variadic elements off the stack.
  SmallVector<TypedAttr> values;
  for (unsigned i = 0; i != numElems; ++i) {
    ErrorOr<TypedAttr> value =
        state.readAttributeFromMemory(baseAddr + i * *allocSize, elemType);
    if (value)
      return value.takeError();
    values.push_back(value.takeValue());
  }

  return ParamListAttr::get(values, *this);
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Try to narrow all the given type expressions to MLIR types.
static LogicalResult resolveTypes(ArrayRef<TypedAttr> types,
                                  SmallVectorImpl<Type> &resolvedTypes) {
  for (const TypedAttr &type : types) {
    if (auto constant = dyn_cast<TypeParamAttr>(type))
      resolvedTypes.push_back(constant.getMlirType());
    else
      return failure();
  }
  return success();
}

/// Parse the StructType.
/// Format:
///   - concrete: `<` `(` types `)` [ `memoryOnly` ] `>`
///   - parametric: `<` param_value [ `memoryOnly` ] `>`
/// Where concrete types can be:
///   - empty: `()`
///   - concrete types: `(i32, i64)`
Type StructType::parse(AsmParser &p) {
  if (p.parseLess())
    return {};

  TypedAttr variadic;

  // Check for `(` which indicates concrete types.
  if (succeeded(p.parseOptionalLParen())) {
    auto metatype = TypeType::get(p.getContext());
    auto variadicType = ParamListType::get(metatype);
    // Check for empty struct case.
    if (succeeded(p.parseOptionalRParen())) {
      // Empty struct.
      variadic = ParamListAttr::get({}, variadicType);
    } else {
      // Parse concrete types.
      SmallVector<Type> types;
      if (parseParamTypes(p, types))
        return {};
      // Create TypeParamAttrs for each element type.
      SmallVector<TypedAttr> elements;
      elements.reserve(types.size());
      for (Type type : types)
        elements.push_back(cast<TypedAttr>(TypeParamAttr::get(type, metatype)));
      variadic = ParamListAttr::get(elements, variadicType);

      if (p.parseRParen())
        return {};
    }
  } else {
    Type variadicType;
    if (succeeded(p.parseOptionalColon())) {
      if (parseKGENType(p, variadicType))
        return {};
    } else {
      variadicType = ParamListType::get(TypeType::get(p.getContext()));
    }

    // Parametric case - parse param value with implicit param_list<!kgen.type>.
    if (parseParamValue(p, variadic, variadicType))
      return {};
  }

  // Parse optional isMemoryOnly: "memoryOnly" or "memoryOnly(<expr>)".
  TypedAttr isMemoryOnly;
  if (failed(KGEN::parseIsMemoryOnly(p, isMemoryOnly)))
    return {};

  bool isParamPack = succeeded(p.parseOptionalKeyword("isParamPack"));

  // Parse optional `align(N)` for minAlignment.
  TypedAttr minAlignment;
  if (succeeded(p.parseOptionalKeyword("align"))) {
    if (p.parseLParen() || parseIndexParamValue(p, minAlignment) ||
        p.parseRParen())
      return {};
  }

  if (p.parseGreater())
    return {};

  return StructType::get(p.getContext(), variadic, isMemoryOnly, minAlignment,
                         isParamPack);
}

/// Print the StructType.
void StructType::print(AsmPrinter &p) const {
  p << '<';

  TypedAttr variadic = getElementTypesVariadic();
  auto attr = dyn_cast<ParamListAttr>(variadic);
  if (!attr || !isa<TypeType>(attr.getType().getElementType())) {
    // Parametric expression or complex metatype - print without parens.  We
    // print :param_list<Movable> if the elements are not TypeType metatype.
    if (!isa<TypeType>(
            cast<ParamListType>(variadic.getType()).getElementType()))
      printColonTypeParamValue(p, variadic);
    else
      printParamValue(p, variadic, variadic.getType());
  } else {
    // Concrete types with simple metatype - print with parens.
    p << '(';
    if (!attr.getValues().empty()) {
      SmallVector<Type> types;
      for (TypedAttr value : attr.getValues())
        types.push_back(ParamType::get(value));
      printParamTypes(p, types);
    }
    p << ')';
  }

  KGEN::printIsMemoryOnly(p, getIsMemoryOnly());

  if (getIsParamPack())
    p << " isParamPack";

  // Print alignment if not the default (1).
  TypedAttr minAlign = getMinAlignment();
  if (auto intAttr = dyn_cast<IntegerAttr>(minAlign)) {
    if (intAttr.getInt() != 1)
      p << " align(" << intAttr.getInt() << ")";
  } else if (minAlign) {
    // Parametric alignment expression.
    p << " align(";
    printIndexParamValue(p, minAlign);
    p << ")";
  }

  p << '>';
}

/// Verify that the variadic parameter is valid.
/// For concrete structs, the variadic should be a ParamListAttr.
/// For parametric structs, it can be any TypedAttr representing a type
/// expression.
LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr variadic, TypedAttr isMemoryOnly,
                                 TypedAttr minAlignment, bool isParamPack) {
  // isMemoryOnly must be an i1-typed attribute (BoolAttr or constraint
  // proposition).
  if (auto intTy = dyn_cast<IntegerType>(isMemoryOnly.getType());
      !intTy || intTy.getWidth() != 1)
    return emitError() << "isMemoryOnly must be i1-typed, but got "
                       << isMemoryOnly.getType();
  // Alignment must have index type.
  if (!sugarIsa<IndexType>(minAlignment.getType()))
    return emitError() << "alignment must have index type, but got "
                       << minAlignment.getType();
  if (isParamPack) {
    if (!isa<IntegerAttr>(isMemoryOnly) ||
        cast<IntegerAttr>(isMemoryOnly).getInt() != 0)
      return emitError() << "packs must not be memory-only";
  }

  // Accept any ParamListType (for concrete cases).
  if (llvm::isa<ParamListType>(variadic.getType()))
    return success();
  // Accept any TypeType-based expression (for parametric single-type cases).
  if (llvm::isa<TypeType>(variadic.getType()))
    return success();
  return emitError() << "expected an operand of variadic or type type, but got "
                     << variadic.getType();
}

bool StructType::isDefinitelyMemoryOnly() const {
  if (auto boolAttr = dyn_cast<BoolAttr>(getIsMemoryOnly()))
    return boolAttr.getValue();
  // i1 constraint proposition (conditional RegisterPassable): not yet folded to
  // BoolAttr — callers that must pick a single ABI (e.g. lowering before
  // elaboration completes) must assume memory-only.
  return true;
}

bool StructType::isNoneOrEmpty(Type type) {
  if (llvm::isa<NoneType>(type))
    return true;
  if (auto structTy = dyn_cast<StructType>(type)) {
    auto resolved = structTy.getVariadicIfResolved();
    if (resolved && resolved.getValues().empty() &&
        !structTy.isDefinitelyMemoryOnly())
      return true;
  }
  return false;
}

ParamListAttr StructType::getVariadicIfResolved() const {
  return dyn_cast_if_present<ParamListAttr>(getElementTypesVariadic());
}

bool StructType::isResolved() const {
  return getVariadicIfResolved() != nullptr;
}

std::optional<SmallVector<Type>> StructType::getElementTypes() const {
  ParamListAttr resolved = getVariadicIfResolved();
  if (!resolved)
    return std::nullopt;

  SmallVector<Type> types;
  types.reserve(resolved.getValues().size());
  for (TypedAttr value : resolved.getValues())
    types.push_back(ParamType::get(value));
  return types;
}

std::optional<size_t> StructType::getNumElements() const {
  ParamListAttr resolved = getVariadicIfResolved();
  if (!resolved)
    return std::nullopt;
  return resolved.getValues().size();
}

static std::optional<int64_t> getPackedElementsTypeSize(ArrayRef<Type> types,
                                                        TargetInfoAttr target) {
  int64_t size = 0;
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeAllocSize(target, type);
    if (!typeAlign || !typeSize)
      return {};
    size = llvm::alignTo(size, *typeAlign) + *typeSize;
    strictest = std::max(strictest, *typeAlign);
  }
  return llvm::alignTo(size, strictest);
}

std::optional<int64_t> StructType::getTypeSize(TargetInfoAttr target) const {
  // A struct backed by a concrete ParamListAttr has a known size.
  if (ParamListAttr attr = getVariadicIfResolved()) {
    SmallVector<Type> types;
    if (failed(resolveTypes(attr.getValues(), types)))
      return {};
    return getPackedElementsTypeSize(types, target);
  }
  // Parametric struct - size unknown until elaboration.
  return {};
}

static std::optional<int64_t> getMaxAlignmentAmongTypes(ArrayRef<Type> types,
                                                        TargetInfoAttr target) {
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!typeAlign)
      return {};
    strictest = std::max(strictest, *typeAlign);
  }
  return strictest;
}

std::optional<int64_t> StructType::getTypeAlign(TargetInfoAttr target) const {
  std::optional<int64_t> naturalAlign;

  // A struct backed by a concrete ParamListAttr has known alignment.
  ParamListAttr attr = getVariadicIfResolved();
  if (!attr)
    return std::nullopt;

  SmallVector<Type> types;
  if (failed(resolveTypes(attr.getValues(), types)))
    return {};
  naturalAlign = getMaxAlignmentAmongTypes(types, target);
  if (!naturalAlign)
    return std::nullopt;

  // Respect explicit minimum alignment from @align(N) decorator.
  // minAlignment is a TypedAttr to support future parametric alignment.
  // For now, it should always be a concrete IntegerAttr with default value 1.
  auto intAttr = dyn_cast<IntegerAttr>(getMinAlignment());
  assert(intAttr && "minAlignment should be IntegerAttr until parametric "
                    "alignment is implemented");
  return std::max(*naturalAlign, intAttr.getInt());
}

ErrorOrSuccess StructType::writeTo(TypedAttr value, int64_t addr,
                                   InterpreterState &state) const {
  int64_t offset = 0;
  if (auto sv = dyn_cast_if_present<StructAttr>(value)) {
    for (TypedAttr value : sv.getValues()) {
      auto dl = ::cast<DataLayoutInterface>(value.getType());
      // Store each element spaced apart by padding according to its alignment.
      std::optional<int64_t> align = dl.getTypeAlign(state.getTarget());
      std::optional<int64_t> size = dl.getTypeSize(state.getTarget());
      if (!align || !size)
        return Error(kStructElemLayoutError);
      offset = llvm::alignTo(offset, *align);

      ErrorOrSuccess result =
          state.writeAttributeToMemory(addr + offset, value);
      if (result.isError())
        return result.takeError();
      offset += *size;
    }
    return success();
  }

  return Error("struct not a writeable type, got " + mlir::debugString(value) +
               " instead");
}

ErrorOr<TypedAttr> StructType::readFrom(int64_t addr,
                                        InterpreterState &state) const {
  std::optional<SmallVector<Type>> elementTypes = getElementTypes();
  if (!elementTypes)
    return Error("cannot read from parametric struct type");
  SmallVector<TypedAttr> values;
  int64_t offset = 0;
  for (Type elType : *elementTypes) {
    auto dl = llvm::cast<DataLayoutInterface>(elType);
    std::optional<int64_t> align = dl.getTypeAlign(state.getTarget());
    std::optional<int64_t> size = dl.getTypeSize(state.getTarget());
    if (!align || !size)
      return Error(kStructElemLayoutError);
    offset = llvm::alignTo(offset, *align);
    ErrorOr<TypedAttr> value =
        state.readAttributeFromMemory(addr + offset, elType);
    if (value.isError())
      return value.takeError();
    values.push_back(value.takeValue());
    offset += *size;
  }
  return StructAttr::get(values, *this);
}

/// Convert ArrayRef<Type> to ParamListAttr for the struct representation.
///
/// Each element type is wrapped in a TypedAttr via TypeParamAttr::get().
/// This may return different attr kinds (TypeParamAttr or ParamOperatorAttr)
/// depending on the input type, but the canonicalization is deterministic
/// so identical input types always produce identical attrs, ensuring
/// consistent type uniquing.
static TypedAttr convertTypesToVariadicAttr(MLIRContext *context,
                                            ArrayRef<Type> types) {
  auto metatype = TypeType::get(context);
  auto variadicType = ParamListType::get(metatype);
  SmallVector<TypedAttr> elements;
  elements.reserve(types.size());
  for (Type type : types)
    elements.push_back(cast<TypedAttr>(TypeParamAttr::get(type, metatype)));
  return ParamListAttr::get(elements, variadicType);
}

/// Normalize a ParamListAttr to use canonicalized TypeParamAttrs.
/// This ensures consistent uniquing regardless of how the ParamListAttr was
/// created (e.g., from bytecode vs from ArrayRef<Type>).
static TypedAttr normalizeVariadicForUniquing(MLIRContext *context,
                                              TypedAttr variadic) {
  // Only normalize concrete VariadicAttrs, not parametric expressions.
  auto variadicAttr = dyn_cast<ParamListAttr>(variadic);
  if (!variadicAttr)
    return variadic;

  // Directly canonicalize TypeParamAttr values by rebuilding them through
  // TypeParamAttr::get(), which ensures consistent representation.
  // Non-TypeParamAttr values (parametric expressions) are kept as-is.
  auto metatype = TypeType::get(context);
  SmallVector<TypedAttr> normalized;
  normalized.reserve(variadicAttr.getValues().size());
  bool changed = false;
  for (TypedAttr value : variadicAttr.getValues()) {
    if (auto typeParam = dyn_cast<TypeParamAttr>(value)) {
      // Canonicalize by rebuilding through TypeParamAttr::get().
      TypedAttr canonical = cast<TypedAttr>(
          TypeParamAttr::get(typeParam.getMlirType(), metatype));
      normalized.push_back(canonical);
      if (canonical != value)
        changed = true;
    } else {
      // Keep parametric expressions as-is.
      normalized.push_back(value);
    }
  }

  // Return original if nothing changed.
  if (!changed)
    return variadic;

  return ParamListAttr::get(normalized, variadicAttr.getType());
}

/// Main builder for StructType with TypedAttr variadic.
/// If minAlignment is null, uses the default value of 1.
/// If isMemoryOnly is null, uses BoolAttr(false).
StructType StructType::get(MLIRContext *context, TypedAttr variadic,
                           TypedAttr isMemoryOnly, TypedAttr minAlignment,
                           bool isParamPack) {
  if (!isMemoryOnly)
    isMemoryOnly = BoolAttr::get(context, false);
  if (!minAlignment)
    minAlignment = IntegerAttr::get(IndexType::get(context), 1);
  if (!isParamPack)
    variadic = normalizeVariadicForUniquing(context, variadic);
  return Base::get(context, variadic, isMemoryOnly, minAlignment, isParamPack);
}

StructType StructType::get(ArrayRef<Type> types, bool isMemoryOnly) {
  assert(!types.empty() &&
         "cannot infer context from empty types; "
         "use get(MLIRContext*, ArrayRef<Type>, bool) for empty structs");
  MLIRContext *context = types.front().getContext();
  TypedAttr variadic = convertTypesToVariadicAttr(context, types);
  return StructType::get(context, variadic,
                         BoolAttr::get(context, isMemoryOnly));
}

StructType StructType::get(MLIRContext *context, ArrayRef<Type> types,
                           bool isMemoryOnly, TypedAttr minAlignment) {
  TypedAttr variadic = convertTypesToVariadicAttr(context, types);
  return StructType::get(context, variadic,
                         BoolAttr::get(context, isMemoryOnly), minAlignment);
}

StructType StructType::get(MLIRContext *context, ArrayRef<Type> types,
                           TypedAttr isMemoryOnly, TypedAttr minAlignment) {
  TypedAttr variadic = convertTypesToVariadicAttr(context, types);
  return StructType::get(context, variadic, isMemoryOnly, minAlignment);
}

StructType StructType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<Type> types, bool isMemoryOnly) {
  assert(!types.empty() &&
         "cannot infer context from empty types; "
         "use getChecked with explicit context for empty structs");
  return getChecked(emitError, types.front().getContext(), types, isMemoryOnly);
}

StructType StructType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<Type> types,
                                  bool isMemoryOnly) {
  TypedAttr variadic = convertTypesToVariadicAttr(context, types);
  TypedAttr isMemOnlyAttr = BoolAttr::get(context, isMemoryOnly);
  TypedAttr minAlignment = IntegerAttr::get(IndexType::get(context), 1);
  if (failed(verify(emitError, variadic, isMemOnlyAttr, minAlignment, false)))
    return {};
  return get(context, variadic, isMemOnlyAttr, minAlignment);
}

StructType StructType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, TypedAttr variadic,
                                  TypedAttr isMemoryOnly,
                                  TypedAttr minAlignment, bool isParamPack) {
  if (!isMemoryOnly)
    isMemoryOnly = BoolAttr::get(context, false);
  if (!minAlignment)
    minAlignment = IntegerAttr::get(IndexType::get(context), 1);
  if (!isParamPack)
    variadic = normalizeVariadicForUniquing(context, variadic);
  if (failed(
          verify(emitError, variadic, isMemoryOnly, minAlignment, isParamPack)))
    return {};
  return Base::get(context, variadic, isMemoryOnly, minAlignment, isParamPack);
}

FailureOr<Type>
StructType::evaluateWithContext(ParameterEvaluationContext &context) const {
  // Perform validation only in materialization contexts.
  if (!context.isMaterializationContext())
    return failure();

  // Validate the alignment value if it's a concrete integer.
  TypedAttr minAlign = getMinAlignment();
  auto intAttr = dyn_cast<IntegerAttr>(minAlign);
  if (!intAttr) {
    // Parametric alignment is not expected in materialization contexts.
    context.emitMaterializationError(
        "alignment did not fold to a concrete integer by materialization");
    return failure();
  }

  int64_t alignVal = intAttr.getInt();

  // Shortcut for default alignment of 1.
  if (alignVal == 1)
    return failure();

  // Validate: must be positive power of 2.
  if (alignVal <= 0 || !llvm::isPowerOf2_64(alignVal)) {
    context.emitMaterializationError(
        "struct alignment must be a positive power of 2, got " +
        Twine(alignVal));
    return failure();
  }

  // Validate: must not exceed reasonable upper bound (2^29 bytes = 512MB).
  // This matches common compiler limits and avoids overflow issues.
  constexpr int64_t kMaxAlignment = 1LL << 29;
  if (alignVal > kMaxAlignment) {
    context.emitMaterializationError(
        "struct alignment exceeds maximum alignment (2^29), got " +
        Twine(alignVal));
    return failure();
  }

  // Validation passed, no folding needed. Return failure to indicate
  // no change was made.
  return failure();
}

//===----------------------------------------------------------------------===//
// StructInstanceType
//===----------------------------------------------------------------------===//

StructInstanceType StructInstanceType::get(StringAttr name,
                                           ArrayRef<StringAttr> paramNames,
                                           ArrayRef<TypedAttr> paramValues,
                                           ArrayRef<StructDefFieldAttr> fields,
                                           TypedAttr isMemoryOnly) {
  return get(name.getContext(), name, paramNames, paramValues, fields,
             isMemoryOnly);
}

StructInstanceType StructInstanceType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    ArrayRef<StringAttr> paramNames, ArrayRef<TypedAttr> paramValues,
    ArrayRef<StructDefFieldAttr> fields, TypedAttr isMemoryOnly) {
  if (failed(verify(emitError, name, paramNames, paramValues, fields,
                    isMemoryOnly)))
    return {};
  return get(name.getContext(), name, paramNames, paramValues, fields,
             isMemoryOnly);
}

LogicalResult StructInstanceType::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    ArrayRef<StringAttr> paramNames, ArrayRef<TypedAttr> paramValues,
    ArrayRef<StructDefFieldAttr> fields, TypedAttr isMemoryOnly) {
  if (paramNames.size() != paramValues.size()) {
    return emitError()
           << "parameter name and parameter value length mismatch. Expected "
           << paramNames.size() << ", got " << paramValues.size();
  }
  // isMemoryOnly must be an i1-typed attribute (BoolAttr or constraint
  // proposition).
  if (auto intTy = dyn_cast<IntegerType>(isMemoryOnly.getType());
      !intTy || intTy.getWidth() != 1)
    return emitError() << "isMemoryOnly must be i1-typed, but got "
                       << isMemoryOnly.getType();
  return success();
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

static ParseResult parseVariantTypes(AsmParser &p, TypedAttr &variadic) {
  auto metatype = TypeType::get(p.getContext());
  auto variadicType = ParamListType::get(metatype);

  // Special case `[<variadic>]` to parse the variadic parameter directly.
  if (succeeded(p.parseOptionalLSquare())) {
    if (parseParamValue(p, variadic, variadicType))
      return failure();
    return p.parseRSquare();
  }

  // Otherwise, parse a non-empty list of concrete types.
  SmallVector<Type> values;
  if (parseParamTypes(p, values))
    return failure();
  SmallVector<TypedAttr> elements;
  for (Type type : values)
    elements.push_back(TypeParamAttr::get(type, metatype));
  variadic = ParamListAttr::get(elements, variadicType);
  return success();
}

static void printVariantTypes(AsmPrinter &p, TypedAttr variadic) {
  // Print an empty variadic or a parametric variadic.
  auto attr = dyn_cast<ParamListAttr>(variadic);
  if (!attr || attr.getValues().empty()) {
    p << '[';
    printParamValue(p, variadic);
    p << ']';
    return;
  }

  SmallVector<Type> values;
  for (TypedAttr value : attr.getValues())
    values.push_back(ParamType::get(value));
  printParamTypes(p, values);
}

OptionalParseResult VariantType::parseValue(AsmParser &p,
                                            TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return {};
  TypedAttr element;
  unsigned index;
  if (parseColonTypeParamValue(p, element) || p.parseComma() ||
      p.parseInteger(index) || p.parseRBrace())
    return failure();
  value = VariantAttr::get(element, index, *this);
  return mlir::success();
}

LogicalResult VariantType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto variant = dyn_cast<VariantAttr>(value);
  if (!variant)
    return failure();
  p << '{';
  printColonTypeParamValue(p, variant.getValue());
  p << ", " << variant.getIndex() << '}';
  return success();
}

VariantType VariantType::get(MLIRContext *ctx, TypedAttr variadic) {
  // When the type is concrete, canonicalize away the type info.
  if (auto attr = dyn_cast<ParamListAttr>(variadic)) {
    SmallVector<TypedAttr> values;
    auto metatype = TypeType::get(ctx);
    for (TypedAttr value : attr.getValues()) {
      values.push_back(TypeParamAttr::get(ParamType::get(value), metatype));
    }
    variadic = ParamListAttr::get(values, ParamListType::get(metatype));
  }
  return Base::get(ctx, variadic);
}

VariantType VariantType::get(ArrayRef<Type> types) {
  assert(!types.empty());
  SmallVector<TypedAttr> values;
  MLIRContext *ctx = types.front().getContext();
  auto metatype = TypeType::get(ctx);
  for (Type type : types)
    values.push_back(TypeParamAttr::get(type, metatype));
  return get(ctx, ParamListAttr::get(values, ParamListType::get(metatype)));
}

VariantType VariantType::getFromBytecode(TypedAttr variadic) {
  return Base::get(variadic.getContext(), variadic);
}

llvm::iterator_range<
    llvm::mapped_iterator<ArrayRef<TypedAttr>::iterator, Type (*)(TypedAttr)>>
VariantType::getTypes() const {
  auto attr = dyn_cast<ParamListAttr>(getVariadic());
  assert(attr && "expected a concrete variant");
  return llvm::map_range(
      attr.getValues(),
      +[](TypedAttr attr) -> Type { return ParamType::get(attr); });
}

size_t VariantType::getNumTypes() const { return llvm::size(getTypes()); }

Type VariantType::getType(unsigned index) const {
  return *std::next(getTypes().begin(), index);
}

/// Compute the size in bytes of just the content section of a variant. The
/// content field is the biggest element size rounded up to the nearest
/// multiple of the content element type size, which is i64.
std::optional<int64_t>
VariantType::getContentSize(TargetInfoAttr target) const {
  auto variadic = dyn_cast<ParamListAttr>(getVariadic());
  if (!variadic)
    return {};

  int64_t maxSize = 0;
  for (TypedAttr value : variadic.getValues()) {
    Type elType = ParamType::get(value);
    // FIXME: Here and above: This seems to be a misuse of API, we should
    // probably use DataLayoutInterface::getTypeStoreSize
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeAllocSize(target, elType);
    if (!typeSize)
      return {};
    maxSize = std::max(maxSize, *typeSize);
  }
  // We could not properly determine the alignment here without taking
  // discriminator into account.
  return maxSize;
}

/// Get bitwidth of the integer used to represent the discriminator. The
/// discriminator field is the smallest integer type whose maximum value is
/// greater than the number of possible subtypes, but which is at least `i1`.
/// The size is rounded to the nearest integer type with a power of 2 bytewidth.
size_t VariantType::getDiscrSizeInBits() const {
  if (!getNumTypes())
    return CHAR_BIT;
  // Compute the smallest iN where N > 0 that fits the count.
  uint64_t bits = std::max(1u, llvm::Log2_32_Ceil(getNumTypes()));
  // Now ceildiv this to the nearest byte multiple.
  uint64_t bytes = llvm::divideCeil(bits, CHAR_BIT);
  // Now compute the nearest power of 2 and convert back to bits.
  return CHAR_BIT * llvm::PowerOf2Ceil(bytes);
}

/// Get the width of the integer used to represent the discriminator in bytes.
/// This returns at least 1, because the bitwidth of the discriminator is at
/// least 8.
static int64_t getVariantDiscrSize(VariantType type) {
  return type.getDiscrSizeInBits() / CHAR_BIT;
}

std::optional<int64_t> VariantType::getTypeSize(TargetInfoAttr target) const {
  // A variant is lowered to a struct that consists of a content field and a
  // discriminator field.
  std::optional<int64_t> contentSize = getContentSize(target);
  if (!contentSize)
    return {};
  // Align to the content array element alignment. We don't expect the
  // discriminator to exceed it in size (at least a 32-bit integer).
  std::optional<int64_t> discrAlign = DataLayoutInterface::getTypeABIAlign(
      target, IntegerType::get(getContext(), getDiscrSizeInBits()));
  if (!discrAlign)
    return {};

  std::optional<int64_t> align = getTypeAlign(target);
  if (!align)
    return {};
  return llvm::alignTo(llvm::alignTo(*contentSize, *discrAlign) +
                           getVariantDiscrSize(*this),
                       *align);
}

std::optional<int64_t> VariantType::getTypeAlign(TargetInfoAttr target) const {
  auto variadic = dyn_cast<ParamListAttr>(getVariadic());
  if (!variadic)
    return {};

  SmallVector<Type> types =
      llvm::map_to_vector(variadic.getValues(), [](TypedAttr value) {
        return ParamType::get(value);
      });
  types.push_back(IntegerType::get(getContext(), getDiscrSizeInBits()));

  return getMaxAlignmentAmongTypes(types, target);
}

ErrorOrSuccess VariantType::writeTo(TypedAttr value, int64_t addr,
                                    InterpreterState &state) const {
  // Just write the value to the address and then the discriminator.
  if (auto variant = dyn_cast_if_present<VariantAttr>(value)) {
    TypedAttr typeValue = variant.getValue();
    ErrorOrSuccess result = state.writeAttributeToMemory(addr, typeValue);
    if (result.isError())
      return result.takeError();
    addr += *getContentSize(state.getTarget());

    unsigned discrSize = getVariantDiscrSize(*this);
    ErrorOr<void *> mem = state.getWritableMemory(addr, discrSize);
    if (mem.isError())
      return mem.takeError();
    APInt discrVal(discrSize * CHAR_BIT, variant.getIndex());
    llvm::StoreIntToMemory(discrVal, reinterpret_cast<uint8_t *>(*mem),
                           discrSize);
    return success();
  }

  return Error("variant not a writeable type, got " + mlir::debugString(value) +
               " instead");
}

ErrorOr<TypedAttr> VariantType::readFrom(int64_t addr,
                                         InterpreterState &state) const {
  // Read the discriminator first so we know what type to read.
  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<const void *> mem = state.getReadableMemory(
      addr + *getContentSize(state.getTarget()), discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(discrVal, reinterpret_cast<const uint8_t *>(*mem),
                          discrSize);

  unsigned index = discrVal.getZExtValue();
  ErrorOr<TypedAttr> result =
      state.readAttributeFromMemory(addr, getType(index));
  if (result.isError())
    return result.takeError();
  return VariantAttr::get(result.takeValue(), index, *this);
}

//===----------------------------------------------------------------------===//
// DeferredType
//===----------------------------------------------------------------------===//

// Deferred types don't have a runtime representation, but can sometimes get
// exposed to the interpreter (e.g. in stack allocations) when they interact
// with generic code.
std::optional<int64_t> DeferredType::getTypeSize(TargetInfoAttr target) const {
  return 0;
}

std::optional<int64_t> DeferredType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// MLIRDeferredType
//===----------------------------------------------------------------------===//

// Placeholder sizes — must be resolved to a concrete type before code gen.
std::optional<int64_t>
MLIRDeferredType::getTypeSize(TargetInfoAttr target) const {
  return 0;
}

std::optional<int64_t>
MLIRDeferredType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// NeverType
//===----------------------------------------------------------------------===//

// NeverType is lowered to empty struct. In LLVM IR it's zero sized type
std::optional<int64_t> NeverType::getTypeSize(TargetInfoAttr target) const {
  return 0;
}

// NeverType is lowered to empty struct. In LLVM IR it has 1 byte alignment
std::optional<int64_t> NeverType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult SIMDType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr size, TypedAttr dtype) {
  if (!size || !dtype)
    return emitError() << "simd type requires size and dtype";
  if (!size.getType().isIndex())
    return emitError() << "size parameter for simd must have type `index`";
  if (!llvm::isa<DTypeType>(dtype.getType()))
    return emitError() << "type parameter for simd must be a !kgen.dtype";
  return success();
}

std::optional<KGENDType> SIMDType::getResolvedDType() const {
  if (auto dtypeAttr = llvm::dyn_cast<DTypeConstantAttr>(getDType()))
    return dtypeAttr.getDType();
  return {};
}

std::optional<int64_t> SIMDType::getResolvedSize() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

SIMDType SIMDType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

SIMDType SIMDType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              TypedAttr size, TypedAttr dtype) {
  return getChecked(emitError, size.getContext(), size, dtype);
}

SIMDType SIMDType::get(int64_t size, TypedAttr dtype) {
  return get(Builder(dtype.getContext()).getIndexAttr(size), dtype);
}

SIMDType SIMDType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              int64_t size, TypedAttr dtype) {
  return getChecked(emitError, Builder(dtype.getContext()).getIndexAttr(size),
                    dtype);
}

SIMDType SIMDType::get(TypedAttr size, KGENDType dtype) {
  return get(size, DTypeConstantAttr::get(size.getContext(), dtype));
}

SIMDType SIMDType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              TypedAttr size, KGENDType dtype) {
  return getChecked(emitError, size,
                    DTypeConstantAttr::get(size.getContext(), dtype));
}

SIMDType SIMDType::get(MLIRContext *ctx, int64_t size, KGENDType dtype) {
  return get(size, DTypeConstantAttr::get(ctx, dtype));
}

SIMDType SIMDType::getScalarBoolType(MLIRContext *ctx) {
  return get(ctx, /*size=*/1, KGENDType::kBool);
}

SIMDType SIMDType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *ctx, int64_t size, KGENDType dtype) {
  return getChecked(emitError, size, DTypeConstantAttr::get(ctx, dtype));
}

std::optional<int64_t> SIMDType::getTypeSize(TargetInfoAttr target) const {
  std::optional<KGENDType> dtype = getResolvedDType();
  std::optional<int64_t> size = getResolvedSize();
  if (!dtype || !size)
    return {};

  switch (dtype->getValue()) {
  case KGENDType::address:
    return llvm::divideCeil(target.getDataLayout().getPointerBitWidth() * *size,
                            CHAR_BIT);
  case KGENDType::index:
  case KGENDType::uindex:
    return llvm::divideCeil(target.resolveIndexBitWidth() * *size, CHAR_BIT);
  default:
    break;
  }
  ssize_t result = dtype->getSizeInBytes(*size);
  // Return zero size for invalid/nonmaterializable dtypes.
  if (result == -1)
    return 0;
  return result;
}

std::optional<int64_t> SIMDType::getTypeAlign(TargetInfoAttr target) const {
  // FIXME: this is inconsistent with the alignment after lowering to LLVM.
  // e.g., Scalar<Int256> will have 32 byte alignment, but i256 in llvm has 16
  // bytes alignment, this seems fine as the KGEN alignment is larger, but we
  // could end up allocating a larger-than-necessary heap memory. Besides, it
  // also cause value mismatch between `sizeof[T]()` and
  // `unsafe_pointer[T] + 1 - unsafe_pointer[T]`.
  if (std::optional<int64_t> size = getTypeSize(target))
    return std::max((int64_t)llvm::PowerOf2Ceil(*size), (int64_t)1);
  return {};
}

ErrorOrSuccess SIMDType::writeTo(TypedAttr value, int64_t addr,
                                 InterpreterState &state) const {
  KGENDType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem = state.getWritableMemory(addr, vecSize);
  if (mem.isError())
    return mem.takeError();
  auto *data = reinterpret_cast<uint8_t *>(*mem);

  auto sv = dyn_cast_if_present<SIMDAttr>(value);
  if (!sv) {
    return Error("SIMD not a writeable type, got " + mlir::debugString(value) +
                 " instead");
  }

  ArrayRef<DTypeValue> values = sv.getValues();

  // Integer dtypes s/ui1/2/4 are densely packed. Handle them here.
  if (dtype.isInt()) {
    unsigned bitWidth = dtype.getIntegerWidthInBits();
    if (bitWidth < CHAR_BIT) {
      assert(CHAR_BIT % bitWidth == 0);
      for (unsigned i = 0, e = values.size(); i != e;) {
        APInt value(CHAR_BIT, 0);
        for (unsigned j = 0; j != CHAR_BIT && i != e; j += bitWidth, ++i)
          value |= values[i].getIntVal().zext(CHAR_BIT).shl(j);
        llvm::StoreIntToMemory(value, data++, 1);
      }
      return success();
    }
  }

  // Other dtypes are multiples of bytes.
  int64_t byteSize = vecSize / *getResolvedSize();
  for (const DTypeValue &value : values) {
    llvm::StoreIntToMemory(value.getData(), data, byteSize);
    data += byteSize;
  }
  return success();
}

ErrorOr<TypedAttr> SIMDType::readFrom(int64_t addr,
                                      InterpreterState &state) const {
  KGENDType dtype = *getResolvedDType();
  int64_t vecSize = *getTypeSize(state.getTarget());
  ErrorOr<const void *> mem = state.getReadableMemory(addr, vecSize);
  if (mem.isError())
    return mem.takeError();
  auto *data = reinterpret_cast<const uint8_t *>(*mem);
  int64_t count = *getResolvedSize();

  // Integer dtypes s/ui1/2/4 are densely packed. Handle them here.
  if (dtype.isInt()) {
    unsigned bitWidth = dtype.getIntegerWidthInBits();
    if (bitWidth < CHAR_BIT) {
      assert(CHAR_BIT % bitWidth == 0);
      SmallVector<DTypeValue> values;
      for (unsigned i = 0; i != count;) {
        APInt value(CHAR_BIT, 0);
        llvm::LoadIntFromMemory(value, data++, 1);
        for (unsigned j = 0; j != CHAR_BIT && i != count; j += bitWidth, ++i)
          values.emplace_back(value.lshr(j).trunc(bitWidth), dtype);
      }
      return SIMDAttr::get(values, *this);
    }
  }

  // Other dtypes are multiples of bytes in memory.
  int64_t bitWidth = dtype.getWidthInBits(state.getTarget());
  int64_t byteSize = vecSize / *getResolvedSize();
  int64_t shiftBits = byteSize * CHAR_BIT - bitWidth;
  // Use consistent internal storage width for pointer-sized types to avoid
  // bit width mismatches when cross-compiling to targets with different
  // pointer sizes.
  int64_t storageBitWidth =
      (dtype.isIndex() || dtype.isUIndex() || dtype.isAddress())
          ? IndexType::kInternalStorageBitWidth
          : bitWidth;

  SmallVector<DTypeValue> values;
  APInt value(byteSize * CHAR_BIT, 0);
  for (unsigned i = 0; i != count; ++i) {
    llvm::LoadIntFromMemory(value, data + i * byteSize, byteSize);
    if (bitWidth == -1) {
      // dtype width unknown (e.g. address, index).
      values.emplace_back(value, dtype);
    } else {
      // For FloatTF32, right Shift 32 bit data by 13 bits and trunc to 19 bits;
      // other types, lshr and trunc are no ops.
      values.emplace_back(
          value.lshr(shiftBits).trunc(bitWidth).sextOrTrunc(storageBitWidth),
          dtype);
    }
  }
  return SIMDAttr::get(values, *this);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "Mojo/KGENDialect/KGENTypes.cpp.inc"

/// Parse a type registered to this dialect.
/// For most cases we rely on the default `generatedTypeParser`, but we have a
/// special handling for "scalar<t>", which is a syntactic sugar for
/// "simd<1, t>".
Type KGENDialect::parseType(DialectAsmParser &p) const {
  StringRef mnemonic;
  Type genType;
  mlir::OptionalParseResult parseResult =
      generatedTypeParser(p, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Check for sugared keyword parsers (e.g. "scalar").
  if (auto it = typeParseFns.find(mnemonic); it != typeParseFns.end())
    return it->second(p);

  p.emitError(p.getCurrentLocation())
      << "unknown type `" << mnemonic << "` in dialect `" << getNamespace()
      << "`";
  return {};
}

/// Print a type registered to this dialect.
/// For most cases we rely on the default `generatedTypePrinter`, but the
/// printer for "simd<1, t>" sugars it to "scalar<t>".
void KGENDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto it = typePrintFns.find(type.getTypeID()); it != typePrintFns.end()) {
    it->second(p, type);
    return;
  }
  (void)generatedTypePrinter(type, p);
}
