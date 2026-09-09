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
//
// This file declares utility functions primarily for parsing, printing and
// verifying KGEN related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENUTILS_H
#define KGEN_KGENDIALECT_KGENUTILS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/BitVector.h"

#include <optional>
#include <utility>

namespace M::KGEN {
class DeclInterface;
class FuncInterface;
class SIMDType;

/// This type is a bit of a mouthful, add a useful alias for it.
using ExportMap = llvm::MapVector<StringAttr, ExportKind>;

/// Given a module operation, return its exported symbols and aliases.
ExportMap getExportedSymbols(ModuleOp module);

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file. In diagnostics contexts, MLIR and KGEN keywords
/// are not escaped with *"...".
std::string getParamAsString(Attribute value);

/// Print the value as colon type parameter value into a string.
StringAttr getParamTypeAsString(TypedAttr value);

/// Print the type as a KGEN type.
StringAttr getTypeAsString(Type type);

/// Parse a parameter of type kgen.string.
ParseResult parseStringParam(AsmParser &p, TypedAttr &value);

/// Print a parameter of type kgen.string.
void printStringParam(AsmPrinter &p, Operation *, Attribute value);

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult parseKGENType(AsmParser &parser, Type &type);
OptionalParseResult parseOptionalKGENType(AsmParser &parser, Type &type);

/// Try to parse a specific KGEN type.
template <typename T>
ParseResult parseKGENType(AsmParser &parser, T &type) {
  Type value;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parseKGENType(parser, value)))
    return failure();
  if (auto expectedType = dyn_cast<T>(value))
    return type = expectedType, success();
  return parser.emitError(loc, "wrong KGEN type");
}

/// Print `type` using KGEN specific type sugars.
void printKGENType(AsmPrinter &p, Type type);
void printKGENType(raw_ostream &os, Type type);

/// Parse a "colon type" production if present or default to `defaultType` type
/// if not.
ParseResult parseColonTypeOrDefault(AsmParser &parser, Type &type,
                                    Type defaultType);

/// Parse a "colon type" production if present or default to `index` type if
/// not.  This is commonly used in our parameter representation.
ParseResult parseColonTypeOrIndex(AsmParser &parser, Type &type);

/// Print `: <type>` or elide it entirely if type is `defaultType` type.
void printColonTypeOrDefault(AsmPrinter &p, Type type, Type defaultType);

/// Print `: <type>` or elide it entirely if type is an `index` type.
void printColonTypeOrIndex(AsmPrinter &p, Type type);

/// Returns whether the given type could be the type of a KGEN type expression.
bool isTypeExprType(Type type);

/// Returns whether the given attribute is a KGEN type expression.
bool isTypeExpr(TypedAttr attr);

/// If `prop` asserts that its two operands denote the same value -- so that a
/// consumer may substitute either one for the other -- return the pair.
std::optional<std::pair<TypedAttr, TypedAttr>>
getIdentityProposition(TypedAttr prop);

/// Gets the common Modular environment attribute (also known as `-D` defines)
/// for the given compilation context. This includes things like
/// `BUILD_TYPE`, AsyncRT profiling level, etc.
EnvAttr getModularEnvAttr(MLIRContext *ctx, CompilationContext *compileCtx);

/// Extends the module EnvAttr with common Modular environment attribute (also
/// known as `-D` defines) for the given module. This includes things like
/// `BUILD_TYPE`, AsyncRT profiling level, etc. Note that
/// the existing EnvAttr module values take precedence here.
void extendWithModularEnvAttr(ModuleOp moduleOp,
                              CompilationContext *compileCtx);

/// Gets the common Modular environment attribute (also known as `-D` defines)
/// for the given module.
EnvAttr getModuleEnvAttr(ModuleOp moduleOp);

/// Parser & printer for the isMemoryOnly field on struct types. Handles:
///   (absent)          -> BoolAttr(false), register-passable
///   memoryOnly        -> BoolAttr(true), unconditionally memory-only
///   memoryOnly(<expr>)-> i1-typed constraint proposition for conditional RP
void printIsMemoryOnly(AsmPrinter &p, TypedAttr isMemoryOnly);
ParseResult parseIsMemoryOnly(AsmParser &p, TypedAttr &isMemoryOnly);

/// Parser & printer for explicit minimum alignment (e.g., "align(64)").
/// Uses TypedAttr to support future parametric alignment.
void printMinAlignment(AsmPrinter &p, TypedAttr minAlignment);
ParseResult parseMinAlignment(AsmParser &p, TypedAttr &minAlignment);

/// Parser & printer for array of simplified StructDefFieldAttrs.
ParseResult parseStructDefFields(AsmParser &p,
                                 SmallVector<StructDefFieldAttr> &fields);
void printStructDefFields(AsmPrinter &p, ArrayRef<StructDefFieldAttr> fields);

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//===----------------------------------------------------------------------===//

ParseResult parseBindParams(AsmParser &p, TypedAttr &generator,
                            SmallVectorImpl<TypedAttr> &paramValues,
                            DenseBoolArrayAttr &discharged,
                            Type preParsedGeneratorType = {});
void printBindParams(AsmPrinter &p, TypedAttr generator,
                     ArrayRef<TypedAttr> paramValues,
                     DenseBoolArrayAttr discharged = {});

/// Convert a bitvector mask to a dense bool array attribute. Empty masks are
/// represented as a null attribute so callers keep the compact default form.
DenseBoolArrayAttr getDenseBoolArrayAttr(MLIRContext *context,
                                         const llvm::BitVector &mask);

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise. When
/// printing a parameter name in a reference, the name must be escaped to
/// prevent collision with other parameter values, particularly types.
void printParamName(AsmPrinter &p, StringAttr name, bool isRef = false);
/// Parse a parameter name as either a keyword or double quoted string.
ParseResult parseParamName(AsmParser &p, StringAttr &name);

/// Print & parse a list of parameter names.
void printParamNames(AsmPrinter &p, ArrayRef<StringAttr> names,
                     bool isRef = false);
ParseResult parseParamNames(AsmParser &p, SmallVector<StringAttr> &names);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok. In a
/// context where printing for diagnostics, we do not use the double quoted
/// syntax to escape MLIR and KGEN keywords.
void printParamValue(AsmPrinter &p, TypedAttr value, Type type = {});
void printParamValue(AsmPrinter &p, Operation *op, TypedAttr value,
                     Type type = {});

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(AsmParser &p, TypedAttr &value, Type type,
                            bool disableTypeParser = false);

/// Parse a parameter declaration of the form `name = value`.
ParseResult parseParamDeclaration(OpAsmParser &p, ParamDeclAttr &paramDecl,
                                  TypedAttr &value);

/// Print a parameter declaration of the form `name = value`.
void printParamDeclaration(OpAsmPrinter &p, ParamDeclAttr paramDecl,
                           TypedAttr value);

/// Parse ":type 42" or "42" and default to index type.
ParseResult parseParamValueDefaultingToIndex(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `dtype` type.
void printDTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `dtype` type.
ParseResult parseDTypeParamValue(AsmParser &p, TypedAttr &value);

/// Print a type parameter value. Default to `TypeType`, but allow an
/// optional type for the type value.
void printTypeParamValue(AsmPrinter &p, TypedAttr value);
/// Parse a type parameter value. Prints the type of the type value if it is not
/// an `TypeType`.
ParseResult parseTypeParamValue(AsmParser &p, TypedAttr &value);

/// Parse or print a parametric type expression and convert it to a type.
ParseResult parseParamType(AsmParser &p, Type &type);
void printParamType(AsmPrinter &p, Type type);
ParseResult parseParamTypes(AsmParser &p, SmallVectorImpl<Type> &types);
void printParamTypes(AsmPrinter &p, ArrayRef<Type> types);

/// Print an array of parameter type values.
void printTypeParamValues(AsmPrinter &p, ArrayRef<TypedAttr> values);
/// Parse an array of parameter type values.
ParseResult parseTypeParamValues(AsmParser &p, SmallVector<TypedAttr> &values);

/// Print a trait symbol
void printTraitSymbol(AsmPrinter &p, TraitSymbolAttr trait);
/// Parse a trait symbol
ParseResult parseTraitSymbol(AsmParser &p, TraitSymbolAttr &trait);

/// Print a comma-separated list of trait symbols.
void printTraitSymbols(AsmPrinter &p, ArrayRef<TraitSymbolAttr> traits);
/// Parse a comma-separated list of trait symbols.
ParseResult parseTraitSymbols(AsmParser &p,
                              SmallVectorImpl<TraitSymbolAttr> &traits);

/// Print the body of a type-value (without any surrounding brackets). Caller
/// specifies how types are printed.
void printTypeValueBody(
    AsmPrinter &p, TypeParamAttr type,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter);
/// Parse the body of a type-value (without any surrounding brackets). Caller
/// specifies how types are parsed.
/// If the caller knows the type has identical type-value representation, it
/// can set the additional flag to abort after the first type is parsed.
OptionalParseResult parseTypeValueBody(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser,
    bool knownIdenticalRepresentation = false);

/// Pretty print a type-value:
/// If the type-value has identical type/value representation, just print the
/// type-value Type itself. Otherwise print the entire type-value surrounded by
/// square brackets.
LogicalResult
printSugaredTypeValue(AsmPrinter &p, TypedAttr value,
                      llvm::function_ref<void(AsmPrinter &, Type)> typePrinter);
/// Parse a pretty-printed type-value.
OptionalParseResult parseSugaredTypeValue(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser);

/// Print a parameter value that is known to have `index` type.
void printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `index` type.
ParseResult parseIndexParamValue(AsmParser &p, TypedAttr &value);

/// Parse/print an optional i1 flag in assembly format, e.g. `volatile<1>`.
ParseResult parseI1Flag(AsmParser &p, TypedAttr &value,
                        llvm::StringRef keyword);
void printI1Flag(AsmPrinter &p, TypedAttr value, llvm::StringRef keyword);
inline void printI1Flag(AsmPrinter &p, Operation *, TypedAttr value,
                        llvm::StringRef keyword) {
  printI1Flag(p, value, keyword);
}

/// Print a parameter value that is known to have `scalar<bool>` type.
void printScalarBoolParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printScalarBoolParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `scalar<bool>` type.
ParseResult parseScalarBoolParamValue(AsmParser &p, TypedAttr &value);

/// Parse a index-or-colon-type and then a parameter value of that type.
ParseResult parseColonTypeParamValue(AsmParser &p, TypedAttr &value);
void printColonTypeParamValue(AsmPrinter &p, TypedAttr value);
inline void printColonTypeParamValue(AsmPrinter &p, Operation *,
                                     TypedAttr value) {
  printColonTypeParamValue(p, value);
}

/// Print the operand of a `TypeConformsToTraitAttr`. The operand is stored as
/// a `param_list<!kgen.type>` value. As a shorthand, when the value is a
/// 1-element `ParamListAttr` literal, print just its upcast-stripped element.
void printConformsToParamList(AsmPrinter &p, TypedAttr typeValue);
/// Parse a `TypeConformsToTraitAttr` operand, accepting either the sugared
/// scalar form or the canonical `param_list<!kgen.type>` storage form.
ParseResult parseConformsToParamList(AsmParser &p, TypedAttr &typeValue);

/// Parse and print a witness entry which has syntactic form `name : type =
/// value`.
ParseResult parseWitnessEntry(AsmParser &p, StringAttr &name,
                              TypedAttr &method);
void printWitnessEntry(AsmPrinter &p, StringAttr name, TypedAttr method);
inline void printWitnessEntry(AsmPrinter &p, Operation *, StringAttr name,
                              TypedAttr method) {
  printWitnessEntry(p, name, method);
}

/// Parse and print a ParamDeclAttr which has syntactic form `name (: type)?`.
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result);
void printParamDecl(AsmPrinter &p, ParamDeclAttr decl);
inline void printParamDecl(AsmPrinter &p, Operation *, ParamDeclAttr decl) {
  printParamDecl(p, decl);
}

/// Type of hooks that customize parameter declaration printing.
using ParamDeclPrintHookTy = function_ref<void(ParamDeclAttr decl)>;

/// Type of hooks that customize parameter declaration parsing.
using ParamDeclParseHookTy =
    function_ref<ParseResult(SmallVectorImpl<ParamDeclAttr> &)>;

/// Parse and print a comma separated list of ParamDeclAttrs.
ParseResult parseParamDeclAttrs(AsmParser &p,
                                SmallVector<ParamDeclAttr> &decls);
void printParamDeclAttrs(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls);

/// Print a ParamDeclArrayAttr as a canonical list of comma separated
/// information. If the element printing hook is provided, it is called by the
/// given parser for each element in the list, and is responsible for printing
/// the decl.
void printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls,
                     ParamDeclPrintHookTy printElt = {});

/// Parse a ParamDeclArrayAttr as a canonical list of comma separated
/// information. If the element parsing hook is provided, it is called by the
/// given parser for each element in the list, and is responsible for parsing
/// the decl and placing it in the provided array.
ParseResult parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result,
                            ParamDeclParseHookTy parseElt = {});

/// Parse and print a parameter specification on a generator or region type. The
/// parameter spec includes input parameter declarations and types and
/// optionally result parameter declarations and types. If the input element
/// parsing hook is provided, it is called by the given parser for each element
/// of the inputs, and is responsible for parsing the decl and placing it in the
/// provided array.
ParseResult parseOptionalParameterSpec(AsmParser &parser,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       ParamDeclArrayAttr &resultParamDecls,
                                       ParamDeclParseHookTy parseInputElt = {});

/// Print a parameter specification on a generator or region type. The parameter
/// spec includes input parameter declarations and types and optionally result
/// parameter declarations and types. If the input element printing hook is
/// provided, it is called by the given parser for each element of the inputs,
/// and is responsible for printing the decl.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> inputParamDecls,
                                ArrayRef<ParamDeclAttr> resultParams = {},
                                ParamDeclPrintHookTy printInputElt = {},
                                ParamDeclPrintHookTy printResultElt = {});

/// Parse an optional argument convention, or use the given default.
ParseResult parseArgConvention(AsmParser &p, ArgConvention &convention);

/// Print an argument convention if not the given default.
void printArgConvention(AsmPrinter &p, ArgConvention convention);

/// Print the parameter type signature if there are any input or result types.
/// If the input type printing hook is provided, it is called by the given
/// parser for each element of the inputs, and is responsible for printing the
/// type.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 function_ref<void(Type)> printInputTy = {});

/// Parse a parameter signature (input/result types) if present. If the input
/// type parsing hook is provided, it is called by the given parser for each
/// element of the inputs, and is responsible for parsing the type and placing
/// it in the provided array.
ParseResult parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    function_ref<ParseResult(SmallVectorImpl<Type> &)> parseInputTy = {});

ParseResult parseSignature(AsmParser &p, TypeAttr &signature);
ParseResult parseSignature(AsmParser &p, Type &signature);
OptionalParseResult parseOptionalSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList);
ParseResult parseSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList);
void printSignature(AsmPrinter &p, Operation *op, TypeAttr signature);
void printSignatureValues(AsmPrinter &p, FunctionType functionType,
                          FuncTypeGeneratorType sigGen);
void printSignatureValues(AsmPrinter &p, function_ref<void(unsigned)> printElt,
                          FunctionType functionType,
                          ArrayRef<ArgConvention> argConvs, FnEffects fnEffects,
                          bool optionalResultList);

/// FuncType versions of print/parse
void printFuncType(AsmPrinter &p, FuncType signatureType);
ParseResult parseFuncType(AsmParser &p, Type &signature);

void printGenerator(AsmPrinter &p, GeneratorType generator);
inline void printGenerator(AsmPrinter &p, Operation *op, Type generator) {
  printGenerator(p, cast<GeneratorType>(generator));
}
ParseResult parseGenerator(AsmParser &p, Type &generator);

/// Parse a plain (i.e. non-lit) func type generator.
ParseResult parseKGENFuncTypeGenerator(AsmParser &p, FunctionType &functionType,
                                       FuncTypeGeneratorType &generator);

/// Parse a function signature with optional metadata. In the assembly format,
/// the SSA value names are optional in the argument list. If they are present,
/// they are populated in `args`. The `parseNames` flag control whether the
/// signature should include the argument names.
ParseResult parseFunctionFuncTypeGenerator(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, FuncTypeGeneratorType &signature,
    ParamDeclParseHookTy parseDeclElt = {});
/// Print a function signature with optional metadata. If `region` is
/// non-null, then the SSA value names of the region arguments are printed.
void printFunctionFuncTypeGenerator(OpAsmPrinter &p, Region *region,
                                    ArrayRef<ParamDeclAttr> inputParams,
                                    ArrayRef<ParamDeclAttr> resultParams,
                                    FunctionType functionType,
                                    FuncTypeGeneratorType signature,
                                    ParamDeclPrintHookTy printInputElt = {},
                                    ParamDeclPrintHookTy printResultElt = {});

/// Parse the always_inline related keywords if present.
ParseResult parseOptionalInline(OpAsmParser &parser, InlineLevelAttr &attr);
void printOptionalInline(AsmPrinter &p, InlineLevel level);

/// Parse and print a decorator list if present.
ParseResult parseOptionalDecorators(AsmParser &p, DecoratorsAttr &decorators);
void printOptionalDecorators(OpAsmPrinter &p, Operation *op,
                             ArrayRef<TypedAttr> decorators);

/// Parse and print a list of parameter values.
ParseResult parseParameterValues(AsmParser &p, ParameterExprArrayAttr &values);
ParseResult parseParameterValues(AsmParser &p,
                                 SmallVectorImpl<TypedAttr> &values);
void printParameterValues(OpAsmPrinter &p, Operation *op,
                          ParameterExprArrayAttr values);
void printParameterValues(AsmPrinter &p, ArrayRef<TypedAttr> values);

/// Parse and print a parametric callee and result parameter declarations.
ParseResult parseParametricCallee(OpAsmParser &p, TypedAttr &callee);
void printParametricCallee(OpAsmPrinter &p, Operation *, TypedAttr callee);

/// Parse and print a comma separated sequence of elements.
template <typename SequenceType>
ParseResult parseSequenceElements(AsmParser &p, SmallVector<TypedAttr> &values,
                                  SequenceType type) {
  return p.parseCommaSeparatedList([&] {
    return parseParamValue(p, values.emplace_back(), type.getElementType());
  });
}

template <typename SequenceType>
void printSequenceElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                           SequenceType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

/// Print and parse an emission kind.
void printEmissionKind(AsmPrinter &p, TypedAttr emissionKind);
ParseResult parseEmissionKind(AsmParser &p, TypedAttr &emissionKind);

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

enum class GeneratorOrFuncKind { func, generator };

/// Parse and print an export kind.
ParseResult parseSymbolExport(AsmParser &p, ExportKindAttr &exportKind);
void printSymbolExport(AsmPrinter &p, Operation *op, ExportKindAttr exportKind);

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult verifyDeclSignaturesMatch(
    StringRef originatorName, FuncTypeGeneratorType originatorSignature,
    Location originatorLoc, StringRef interfaceName,
    FuncTypeGeneratorType targetSignatureGen, Location targetLoc);

/// Check that the specified function types match, checking the parameter and
/// value type information.
LogicalResult verifyFuncTypesMatch(StringRef lhsName, FuncType lhsSigGen,
                                   Location lhsLoc, StringRef rhsName,
                                   FuncType rhsSigGen, Location rhsLoc);

/// Check that the parameter bindings match the declarations.
LogicalResult
verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                      ArrayRef<TypedAttr> paramValues, Location originatorLoc,
                      StringRef targetName, ArrayRef<ParamDeclAttr> decls,
                      Location targetLoc);

/// Verify that the types of operands passed as arguments to a call match the
/// expected types on the callee signature.
LogicalResult verifyCallOperands(Operation *op, ValueRange args,
                                 FuncType callee, bool ignoreByRef = false);
/// Verify that the types of operation results corresponding to call results
/// match the expected types on the callee signature.
LogicalResult verifyCallResults(Operation *op, ValueRange results,
                                FuncType callee);

/// Whether the decorator's name is (starts with) the specific annotation.
bool hasDecorator(ArrayRef<TypedAttr> decorators, StringRef annotation);

/// Whether the generator operation contains any decorator with any of the given
/// annotations.
bool hasAnyDecorator(ArrayRef<TypedAttr> decorators,
                     ArrayRef<StringLiteral> annotations);

ParseResult parseRegionWithArgs(OpAsmParser &p, Region &region);
void printRegionWithArgs(OpAsmPrinter &p, Operation *op, Region &region);

/// Converts the string into a Mojo string literal, making sure the set of
/// special characters supported in Lexer::getStringLiteralValue are taken into
/// account. It is *not* intended to be a general alternative to
/// llvm::printEscapedString.
void printAsMojoStringLiteral(StringRef Name, raw_ostream &Out);

std::string
printSimpleParamAttrValues(ArrayRef<ParamDeclAttr> params,
                           ArrayRef<TypedAttr> values,
                           CompilationOptions::ErrorVerboseLevel verboseLevel);
//===----------------------------------------------------------------------===//
// SIMD Utilities
//===----------------------------------------------------------------------===//

template <uint8_t dtype>
bool isSIMDOf(Type type) {
  if (auto simdType = sugarDynCast<SIMDType>(type))
    return simdType.getResolvedDType() == dtype;
  return false;
}

template <uint8_t dtype>
bool isScalarOf(Type type) {
  if (auto simdType = sugarDynCast<SIMDType>(type))
    return simdType.getResolvedSize() == 1 && isSIMDOf<dtype>(simdType);
  return false;
}

/// Verify a conversion between a SIMD type and an MLIR builtin type.
/// Conversions are assumed to be bi-directional. In error messages, the
/// direction of the conversion is controlled by the `fromSimd` parameter.
LogicalResult
verifyConversionCast(function_ref<InFlightDiagnostic(StringRef)> emitError,
                     SIMDType simd, Type builtinType, bool fromSimd);

/// Fold a CastToBuiltin operation/attribute from a SIMD-typed value to a MLIR
/// builtin type.
OpFoldResult foldCastToBuiltin(TypedAttr input, Type resultType);

/// Fold a conversion from a MLIR builtin type to a SIMD type.
OpFoldResult foldCastFromBuiltin(TypedAttr input, SIMDType resultType);

/// Fold a SIMD splat operation.
OpFoldResult foldSIMDSplat(Value scalarVal, Attribute scalarAttr,
                           SIMDType resultType);

/// Get the equivalent simd scalar type for a given builtin type, return nullptr
/// if there is no equivalent scalar type.
SIMDType getEquivalentSIMDType(Type type);

/// Splat a value of a builtin type to a SIMD type of the given size.
TypedAttr splatBuiltinToSIMD(TypedAttr builtinScalarVal, TypedAttr simdSize);
TypedAttr splatFloatLiteralToSIMD(double literal, SIMDType target);
TypedAttr splatIntLiteralToSIMD(uint64_t literal, SIMDType target);

// Whether all elements of the SIMD attribute are zero/one.
inline bool isAllIntLikeZero(SIMDAttr simdAttr) {
  if (!simdAttr.getType().getResolvedDType()->isIntLike())
    return false;
  return llvm::all_of(simdAttr.getValues(),
                      [](const DTypeValue &v) { return v.getData().isZero(); });
}
inline bool isAllIntLikeOne(SIMDAttr simdAttr) {
  if (!simdAttr.getType().getResolvedDType()->isIntLike())
    return false;
  return llvm::all_of(simdAttr.getValues(),
                      [](const DTypeValue &v) { return v.getData().isOne(); });
}

//===----------------------------------------------------------------------===//
// Constraint Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the proposition is trivially true (constant i1 = 1).
inline bool isTriviallyTrueProposition(TypedAttr prop) {
  auto scalarBoolAttr = dyn_cast<SIMDAttr>(prop);
  return scalarBoolAttr && scalarBoolAttr.getAsBool();
}

/// Returns true if the proposition is trivially true (constant i1 = 1).
inline bool isTriviallyFalseProposition(TypedAttr prop) {
  auto scalarBoolAttr = dyn_cast<SIMDAttr>(prop);
  return scalarBoolAttr && !scalarBoolAttr.getAsBool();
}

/// Returns true if the constraint is trivially true (proposition = constant 1)
/// or null (for backward compatibility with old IR).
inline bool isTriviallyTrueConstraint(ConstraintAttr constraint) {
  if (!constraint)
    return true;
  return isTriviallyTrueProposition(constraint.getProposition());
}

/// Returns true if the constraint is trivially true (proposition = constant 1)
/// or null (for backward compatibility with old IR).
inline bool isTriviallyFalseConstraint(ConstraintAttr constraint) {
  if (!constraint)
    return false;
  return isTriviallyFalseProposition(constraint.getProposition());
}

/// Create a placeholder constraint for unconditional trait conformance.
inline ConstraintAttr getUnconditionalConstraint(MLIRContext *ctx) {
  return ConstraintAttr::get(SIMDAttr::getScalarBool(ctx, true),
                             UnknownLoc::get(ctx), /*message=*/StringAttr());
}

} // namespace M::KGEN

#include "Support/ADT/DenseStringMap.h" // IWYU pragma: keep

#endif // KGEN_KGENDIALECT_KGENUTILS_H
