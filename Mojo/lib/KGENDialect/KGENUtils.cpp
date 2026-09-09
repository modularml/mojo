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
// This file implements utility functions primarily for parsing, printing and
// verifying KGEN related operations and types.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypeInterfaces.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/DType.h"
#include "Support/Preprocessor.h"
#include "Support/STLExtras.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Parameter Type and Value Printing and Parsing
//===----------------------------------------------------------------------===//

std::string KGEN::getParamAsString(Attribute value) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    if (auto ta = dyn_cast<TypedAttr>(value)) {
      StreamAsmPrinter p(os);
      printParamValue(p, ta, {});
    } else {
      os << value;
    }
  }
  return std::string(result.data(), result.size());
}

StringAttr KGEN::getParamTypeAsString(TypedAttr value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  StreamAsmPrinter p(os);
  printColonTypeParamValue(p, value);
  return StringAttr::get(value.getContext(), str);
}

StringAttr KGEN::getTypeAsString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  StreamAsmPrinter p(os);
  printKGENType(p, type);
  return StringAttr::get(type.getContext(), str);
}

/// Parse a parameter of type kgen.string.
ParseResult KGEN::parseStringParam(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, KGEN::StringType::get(p.getContext()));
}

/// Print a parameter of type kgen.string.
void KGEN::printStringParam(AsmPrinter &p, Operation *op, Attribute value) {
  return printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a non-empty parameter list without the surrounding braces.
static ParseResult parseParameterSpec(AsmParser &parser,
                                      ParamDeclArrayAttr &inputParamDecls,
                                      ParamDeclArrayAttr &resultParamDecls,
                                      ParamDeclParseHookTy parseDeclElt) {
  // Parse the input list.
  if (parseParamDecls(parser, inputParamDecls, parseDeclElt))
    return failure();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamDecls(parser, resultParamDecls, parseDeclElt))
      return failure();
  } else {
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  }
  return success();
}

/// Parse an argument or type list with optional metadata. This is an optional
/// parse, which allows the KGEN type parser to check if it is parsing a
/// signature. The provided parseArg hook is responsible for parsing an
/// individual argument and adding its type to the provided array.
OptionalParseResult KGEN::parseOptionalSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList) {
  SmallVector<Type> argTypes, resTypes;

  if (failed(p.parseOptionalLParen()))
    return std::nullopt;
  if (failed(p.parseOptionalRParen())) {
    if (p.parseCommaSeparatedList([&]() { return parseArg(argTypes); }) ||
        p.parseRParen())
      return failure();
  }

  // Parse the function effects. Check for each case to disambiguate the syntax
  // for interfaces.
  auto effectsValue = impl::FnEffects::None;
  StringRef kw;
  while (succeeded(
      p.parseOptionalKeyword(&kw, {"throws", "async", "capturing", "refresult",
                                   "register_passable", "cabi"}))) {
    effectsValue |= *impl::symbolizeFnEffects(kw);

    // No vertical bar? We're done. It's not a parse error, but it does mean we
    // can't specify more effects.
    if (failed(p.parseOptionalVerticalBar()))
      break;
  }

  if (optionalResultList ? p.parseOptionalArrowTypeList(resTypes)
                         : p.parseArrowTypeList(resTypes))
    return failure();

  effects = FnEffects(effectsValue);
  values = p.getBuilder().getFunctionType(argTypes, resTypes);
  return mlir::success();
}

/// Parse an operand and result type list with metadata for a plain (i.e.
/// non-lit) signature.
static OptionalParseResult parseOptionalNewKGENSignature(AsmParser &p,
                                                         Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<ArgConvention> argConventions;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    // Parse the argument type and its input convention.
    if (p.parseType(argTypes.emplace_back()) ||
        parseArgConvention(p, argConventions.emplace_back()))
      return failure();
    return success();
  };

  FunctionType functionType;
  FnEffects effects;
  OptionalParseResult result = parseOptionalSignatureValues(
      p, parseArg, functionType, effects, /*optionalResultList=*/false);
  if (result.has_value() && succeeded(*result)) {
    signature =
        FuncType::getChecked([&] { return p.emitError(loc); }, functionType,
                             argConventions, effects, /*metadata=*/{});
    if (!signature)
      return failure();
  }
  return result;
}

/// Parse a plain (i.e. non-LIT) generator type.
static OptionalParseResult parseOptionalKGENGenerator(AsmParser &p,
                                                      Type &generator) {
  SmallVector<Type> paramTypes;
  Type body;

  bool sawParamList = false;
  if (succeeded(p.parseOptionalLess())) {
    sawParamList = true;
    // A failure is if the param list is not empty, and param type parsing
    // failed.
    if (failed(p.parseOptionalGreater()) &&
        (parseParamTypes(p, paramTypes) || p.parseGreater())) {
      return failure();
    }
  }

  // Try to parse an optional FuncType immediately here because we do not
  // allow standalone FuncTypes yet.
  OptionalParseResult optionalSigBody = parseOptionalNewKGENSignature(p, body);
  if (optionalSigBody.has_value()) {
    if (failed(*optionalSigBody))
      return failure();
    generator = GeneratorType::get(paramTypes, body);
    return mlir::success();
  }

  // For anything that's not a func type generator, require a param list.
  if (!sawParamList)
    return std::nullopt;

  if (parseParamType(p, body))
    return failure();

  generator = GeneratorType::get(paramTypes, body);
  return mlir::success();
}

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
OptionalParseResult KGEN::parseOptionalKGENType(AsmParser &p, Type &type) {
  // Check for sugared types before parsing standard ones. We need to check for
  // each keyword individually, since builtin types are also keywords.
  auto *dialect = p.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot parse KGEN type without KGEN dialect");
  for (auto &[keyword, parseFn] : dialect->typeParseFns) {
    if (p.parseOptionalKeyword(keyword))
      continue;
    type = parseFn(p);
    return failure(!type);
  }

  // Parse symbol references as decl reference types.
  if (dialect->symbolTypeParser) {
    SymbolRefAttr symbol;
    OptionalParseResult result = p.parseOptionalAttribute(symbol);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      FailureOr<Type> symbolResult = dialect->symbolTypeParser(p, symbol);
      if (failed(symbolResult))
        return failure();
      type = *symbolResult;
      return LogicalResult::success();
    }
  }

  // Try to parse an optional generator. Generators begin with `<`.
  {
    GeneratorType generator;
    OptionalParseResult result = parseOptionalKGENGenerator(p, generator);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = generator;
      return LogicalResult::success();
    }
  }

  // Try to parse an optional FuncType. FuncTypes begin with `(`.
  // For now we parse all standalone FuncTypes as FuncType generator types for
  // back-compat.
  {
    FuncType signature;
    OptionalParseResult result = parseOptionalNewKGENSignature(p, signature);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = GeneratorType::get(/*inputParamTypes=*/{}, signature);
      return LogicalResult::success();
    }
  }

  return p.parseOptionalType(type);
}

ParseResult KGEN::parseKGENType(AsmParser &p, Type &type) {
  OptionalParseResult result = parseOptionalKGENType(p, type);
  if (result.has_value())
    return result.value();
  return p.emitError(p.getCurrentLocation(), "expected a KGEN type");
}

void KGEN::printKGENType(raw_ostream &os, Type type) {
  StreamAsmPrinter p(os);
  printKGENType(p, type);
}

void KGEN::printKGENType(AsmPrinter &p, Type type) {
  // Always print an alias if available.
  if (succeeded(p.printAlias(type)))
    return;

  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  auto *dialect = type.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot print KGEN type without KGEN dialect");
  if (auto it = dialect->typePrintFns.find(type.getTypeID());
      it != dialect->typePrintFns.end()) {
    it->second(p, type);
  } else if (auto signature = dyn_cast<FuncType>(type)) {
    printFuncType(p, signature);
  } else if (auto generator = dyn_cast<GeneratorType>(type)) {
    printGenerator(p, generator);
  } else {
    p << type;
  }
}

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return std::nullopt;
  return OptionalParseResult(parseKGENType(parser, type));
}

/// Parse a "colon type" production if present or default to `defaultType` type
/// if not.
ParseResult KGEN::parseColonTypeOrDefault(AsmParser &parser, Type &type,
                                          Type defaultType) {
  auto result = parseOptionalColonType(parser, type);
  if (!result.has_value()) {
    type = defaultType;
    return success();
  }
  return result.value();
}

/// Parse a "colon type" production if present or default to index if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrIndex(AsmParser &parser, Type &type) {
  return parseColonTypeOrDefault(parser, type,
                                 parser.getBuilder().getIndexType());
}

/// print `: <type>` or elide it entirely if type is an `index` type.
void KGEN::printColonTypeOrDefault(AsmPrinter &p, Type type, Type defaultType) {
  // Index type is the default so it doesn't print.
  if (type == defaultType)
    return;
  p << ": ";
  printKGENType(p, type);
}

/// print `: <type>` or elide it entirely if type is an `index` type.
void KGEN::printColonTypeOrIndex(AsmPrinter &p, Type type) {
  return printColonTypeOrDefault(p, type, IndexType::get(type.getContext()));
}

/// print `:<type> ` or elide it entirely if type is an `index` type.
static void printColonTypeOrIndexPrefix(AsmPrinter &p, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  p << ':';
  printKGENType(p, type);
  p << ' ';
}

/// Parse ":type 42" or "42" and default to index type.
ParseResult KGEN::parseParamValueDefaultingToIndex(AsmParser &p,
                                                   TypedAttr &value) {
  Type type = p.getBuilder().getIndexType();
  mlir::OptionalParseResult typePresent = parseOptionalColonType(p, type);
  if (typePresent.has_value() && failed(typePresent.value()))
    return failure();
  return parseParamValue(p, value, type);
}

/// Print a parameter value that is known to have `dtype` type.
void KGEN::printDTypeParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a parameter value that is known to have `dtype` type.
ParseResult KGEN::parseDTypeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, DTypeType::get(p.getContext()));
}

void KGEN::printTypeParamValue(AsmPrinter &p, TypedAttr value) {
  if (!isa<TypeType>(value.getType())) {
    p << ':';
    printKGENType(p, value.getType());
    p << ' ';
  }
  printParamValue(p, value);
}

ParseResult KGEN::parseTypeParamValue(AsmParser &p, TypedAttr &value) {
  Type type;
  if (succeeded(p.parseOptionalColon())) {
    if (parseKGENType(p, type))
      return failure();
  } else {
    type = TypeType::get(p.getContext());
  }
  return parseParamValue(p, value, type);
}

ParseResult KGEN::parseParamType(AsmParser &p, Type &type) {
  OptionalParseResult result = parseOptionalKGENType(p, type);
  if (result.has_value())
    return result.value();

  // If not a mlir Type, it's a parameter in the type-domain. Parse as a
  // parameter and wrap with ParamType.
  TypedAttr typeParam;
  if (parseTypeParamValue(p, typeParam))
    return failure();
  type = ParamType::get(typeParam);
  return success();
}

void KGEN::printParamType(AsmPrinter &p, Type type) {
  // A "ParamType" is either:
  // 1. An actual mlir Type,
  // 2. A type-value in the type domain (i.e. wrapped with ParamType), or
  // 3. A type-value in the value domain (i.e. wrapped with TypeValueType).
  //
  // For case 2, the ParamType wrapper around the internal parameter is
  // omitted for simplicity. The internal parameter is printed directly (with
  // an optional colon type prefix).
  // For case 3, the TypeValueType is NOT omitted to differentiate with case 2.
  if (auto paramRef = dyn_cast<ParamType>(type))
    printTypeParamValue(p, paramRef.getParam());
  else
    printKGENType(p, type);
}

ParseResult KGEN::parseParamTypes(AsmParser &p, SmallVectorImpl<Type> &types) {
  return p.parseCommaSeparatedList(
      [&] { return parseParamType(p, types.emplace_back()); });
}

void KGEN::printParamTypes(AsmPrinter &p, ArrayRef<Type> types) {
  llvm::interleaveComma(types, p, [&](Type type) { printParamType(p, type); });
}

void KGEN::printTypeParamValues(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { printTypeParamValue(p, value); });
}

ParseResult KGEN::parseTypeParamValues(AsmParser &p,
                                       SmallVector<TypedAttr> &values) {
  return p.parseCommaSeparatedList(
      [&] { return parseTypeParamValue(p, values.emplace_back()); });
}

void KGEN::printTraitSymbol(AsmPrinter &p, TraitSymbolAttr trait) {
  p.printAttribute(trait.getSymbol());
  printParameterValues(p, trait.getParamValues());
}

ParseResult KGEN::parseTraitSymbol(AsmParser &p, TraitSymbolAttr &trait) {
  SymbolRefAttr symbol;
  SmallVector<TypedAttr> paramValues;
  if (p.parseAttribute(symbol) || parseParameterValues(p, paramValues))
    return failure();

  trait = TraitSymbolAttr::get(symbol, paramValues);
  return success();
}

void KGEN::printTraitSymbols(AsmPrinter &p, ArrayRef<TraitSymbolAttr> traits) {
  p << '[';
  llvm::interleaveComma(traits, p, [&](TraitSymbolAttr trait) {
    p.printAttribute(trait.getSymbol());
  });
  p << ']';
}

ParseResult KGEN::parseTraitSymbols(AsmParser &p,
                                    SmallVectorImpl<TraitSymbolAttr> &traits) {
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
    TraitSymbolAttr &trait = traits.emplace_back();
    return parseTraitSymbol(p, trait);
  });
}

void KGEN::printTypeValueBody(
    AsmPrinter &p, TypeParamAttr type,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter) {
  typePrinter(p, type.getTypeValue());
  if (type.getMlirType() != type.getTypeValue()) {
    p << ", ";
    typePrinter(p, type.getMlirType());
  }
}

OptionalParseResult KGEN::parseTypeValueBody(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser,
    bool knownIdenticalRepresentation) {
  Type typeValue, mlirType;

  OptionalParseResult result = typeParser(p, typeValue);
  if (!result.has_value())
    return {}; // Not a type-value at all.

  if (failed(*result))
    return failure();

  if (knownIdenticalRepresentation || failed(p.parseOptionalComma())) {
    // This type-value has identical type/value representation. Stop here.
    value = TypeParamAttr::get(typeValue, typeValue, type);
    return mlir::success();
  }

  // Parse the mlirType.
  {
    OptionalParseResult result = typeParser(p, mlirType);
    if (!result.has_value())
      return p.emitError(p.getCurrentLocation(), "expected a type");
    if (failed(*result))
      return failure();
  }

  value = TypeParamAttr::get(typeValue, mlirType, type);
  return mlir::success();
}

LogicalResult KGEN::printSugaredTypeValue(
    AsmPrinter &p, TypedAttr value,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter) {
  auto type = dyn_cast<TypeParamAttr>(value);
  if (!type)
    return failure();

  if (succeeded(p.printAlias(type)))
    return success();

  const bool nonTrivial = !type.hasIdenticalRepresentation();
  if (nonTrivial)
    p << '[';

  printTypeValueBody(p, type, typePrinter);

  if (nonTrivial)
    p << "]";
  return success();
}

OptionalParseResult KGEN::parseSugaredTypeValue(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser) {
  bool nonTrivial = succeeded(p.parseOptionalLSquare());

  OptionalParseResult bodyParseResult = parseTypeValueBody(
      p, value, type, typeParser, /*knownIdenticalRepresentation=*/!nonTrivial);
  if (!bodyParseResult.has_value()) {
    // If a '[' was seen, require a type to be present.
    if (nonTrivial)
      return p.emitError(p.getCurrentLocation(), "expected a type");
    return {};
  }
  if (failed(*bodyParseResult))
    return failure();

  if (nonTrivial && failed(p.parseRSquare()))
    return failure();
  return mlir::success();
}

/// Print/Parse an attribute value that is known to have index type.
void KGEN::printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
ParseResult KGEN::parseIndexParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getIndexType());
}

ParseResult KGEN::parseI1Flag(AsmParser &p, TypedAttr &value,
                              llvm::StringRef keyword) {
  if (failed(p.parseOptionalKeyword(keyword))) {
    value = p.getBuilder().getBoolAttr(false);
    return success();
  }
  if (failed(p.parseLess())) {
    value = p.getBuilder().getBoolAttr(true);
    return success();
  }
  if (failed(parseParamValue(p, value, p.getBuilder().getI1Type())) ||
      failed(p.parseGreater()))
    return failure();
  return success();
}

void KGEN::printI1Flag(AsmPrinter &p, TypedAttr value,
                       llvm::StringRef keyword) {
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    if (boolAttr.getValue())
      p << keyword;
    return;
  }
  p << keyword << '<';
  printParamValue(p, cast<TypedAttr>(value));
  p << '>';
}

/// Print/Parse an attribute value that is known to have scalar<bool> type.
void KGEN::printScalarBoolParamValue(AsmPrinter &p, Operation *op,
                                     Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
void KGEN::printScalarBoolParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
ParseResult KGEN::parseScalarBoolParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value,
                         SIMDType::get(p.getContext(), 1, KGENDType::kBool));
}

ParseResult KGEN::parseColonTypeParamValue(AsmParser &p, TypedAttr &value) {
  Type type;
  if (parseColonTypeOrIndex(p, type) || parseParamValue(p, value, type))
    return failure();

  return success();
}

void KGEN::printColonTypeParamValue(AsmPrinter &p, TypedAttr value) {
  printColonTypeOrIndexPrefix(p, value.getType());
  printParamValue(p, value);
}

ParseResult KGEN::parseWitnessEntry(AsmParser &p, StringAttr &name,
                                    TypedAttr &method) {
  std::string nameStr;
  if (p.parseString(&nameStr))
    return failure();
  name = StringAttr::get(p.getContext(), nameStr);
  Type type;
  if (p.parseColon() || parseKGENType(p, type) || p.parseEqual() ||
      parseParamValue(p, method, type))
    return failure();
  return success();
}

void KGEN::printWitnessEntry(AsmPrinter &p, StringAttr name, TypedAttr method) {
  p.printString(name.getValue());
  p << " : ";
  printKGENType(p, method.getType());
  p << " = ";
  printParamValue(p, method);
}

ParseResult KGEN::parseParamDecl(AsmParser &p, ParamDeclAttr &result) {
  StringAttr name;
  Type type;
  if (parseParamName(p, name) || parseColonTypeOrIndex(p, type))
    return failure();
  result = ParamDeclAttr::get(name, type);
  return success();
}

void KGEN::printParamDecl(AsmPrinter &p, ParamDeclAttr decl) {
  printParamName(p, decl.getName());
  printColonTypeOrIndex(p, decl.getType());
}

ParseResult KGEN::parseParamDeclAttrs(AsmParser &p,
                                      SmallVector<ParamDeclAttr> &decls) {
  return p.parseCommaSeparatedList([&]() {
    decls.emplace_back();
    return parseParamDecl(p, decls.back());
  });
}

void KGEN::printParamDeclAttrs(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls) {
  llvm::interleaveComma(decls, p,
                        [&](ParamDeclAttr decl) { printParamDecl(p, decl); });
}

/// Parse a parameter declaration list if present.
///
///   parameter-decl   ::= identifier (`:` type)?
///   parameter-decl-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
ParseResult KGEN::parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result,
                                  ParamDeclParseHookTy parseElt) {
  auto defaultParseElt = [&](SmallVectorImpl<ParamDeclAttr> &decls) {
    return parseParamDecl(p, decls.emplace_back(ParamDeclAttr()));
  };
  if (!parseElt)
    parseElt = std::move(defaultParseElt);

  // Parse each of the decls.
  SmallVector<ParamDeclAttr> decls;

  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
  } else {
    if (p.parseCommaSeparatedList([&]() { return parseElt(decls); }))
      return failure();
  }

  result = ParamDeclArrayAttr::get(p.getContext(), decls);
  return success();
}

void KGEN::printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls,
                           ParamDeclPrintHookTy printElt) {
  auto defaultPrintElt = [&](ParamDeclAttr decl) { printParamDecl(p, decl); };
  if (!printElt)
    printElt = defaultPrintElt;

  if (decls.empty())
    p << "()";
  else
    llvm::interleaveComma(decls, p, printElt);
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult KGEN::parseOptionalParameterSpec(
    AsmParser &parser, ParamDeclArrayAttr &inputParamDecls,
    ParamDeclArrayAttr &resultParamDecls, ParamDeclParseHookTy parseInputElt) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    inputParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  } else {
    if (parseParameterSpec(parser, inputParamDecls, resultParamDecls,
                           parseInputElt) ||
        parser.parseGreater())
      return failure();
  }
  return success();
}

void KGEN::printOptionalParameterSpec(AsmPrinter &p,
                                      ArrayRef<ParamDeclAttr> inputParamDecls,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      ParamDeclPrintHookTy printInputElt,
                                      ParamDeclPrintHookTy printResultElt) {
  if (inputParamDecls.empty() && resultParams.empty())
    return;

  p << '<';
  printParamDecls(p, inputParamDecls, printInputElt);

  if (!resultParams.empty()) {
    p << " -> ";
    printParamDecls(p, resultParams, printResultElt);
  }
  p << '>';
}

//===----------------------------------------------------------------------===//
// "Pretty" parameter printing and parsing
//===----------------------------------------------------------------------===//

// Parameters are complex nested expressions.  While they have a generic
// printing syntax that is supported in full generality, they often appear in
// tightly controlled situations, e.g. in return operations, in types, or when
// invoking a generator. In these cases we can use a much nicer and more compact
// syntax so we as compiler engineers don't go bonkers looking at IR dumps.

enum class POCAliases : uint32_t {
  // The builtin opcodes have 0...127.
  FIRST_PSEUDO = 128,
  NEG, // negation
  SUB, // subtraction
  NOT,
  NE, // !(==)
  GT, // !(<)
  GE, // !(<=)
  // This is an unknown opcode name.
  kInvalid,
};

/// Returns true if the given string can be represented as a bare identifier.
static bool isLegalMLIRIdentifier(StringRef name) {
  // By making this unsigned, the value passed in to isalnum will always be
  // in the range 0-255. This is important when building with MSVC because
  // its implementation will assert. This situation can arise when dealing
  // with UTF-8 multibyte characters.
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

/// Returns true if the given string could be an MLIR builtin type.
/// TODO: Can't interact directly with the MLIR AsmParser.
static bool isMLIRBuiltinType(StringRef name) {
  // Check for a keyword type.
  static const char *keywordTypes[] = {
      "f4e2m1fn", "f6e2m3fn",   "f6e3m2fn",   "f8e8m0fnu", "f8e5m2", "f8e4m3fn",
      "f8e3m4",   "f8e5m2fnuz", "f8e4m3fnuz", "bf16",      "f16",    "f32",
      "f64",      "f80",        "f128",       "index",     "none"};
  if (llvm::is_contained(keywordTypes, name))
    return true;
  // Check for an integral type: (s|u)*i[0-9]+
  if (name.front() == 's' || name.front() == 'u')
    name = name.drop_front();
  if (name.size() <= 1 || name.front() != 'i')
    return false;
  return llvm::all_of(name.drop_front(), isdigit);
}

ParseResult KGEN::parseParamName(AsmParser &p, StringAttr &name) {
  // If this is a '*'-prefixed double quoted string, then this is an escaped
  // parameter name.
  if (succeeded(p.parseOptionalStar())) {
    std::string value;
    if (failed(p.parseString(&value)))
      return failure();
    name = StringAttr::get(p.getContext(), value);
  } else {
    // Barewords / MLIR keywords are param names otherwise.
    StringRef keyword;
    if (failed(p.parseKeyword(&keyword)))
      return failure();
    name = StringAttr::get(p.getContext(), keyword);
  }
  return success();
}

static ParseResult
parseDischargedBodyConstraints(AsmParser &p, DenseBoolArrayAttr &discharged) {
  if (failed(p.parseOptionalVerticalBar()))
    return success();

  std::string mask;
  if (p.parseString(&mask))
    return failure();

  SmallVector<bool> values;
  values.reserve(mask.size());
  for (char bit : mask) {
    if (bit != '0' && bit != '1') {
      return p.emitError(
          p.getCurrentLocation(),
          "expected discharged mask to contain only '0' and '1'");
    }
    values.push_back(bit == '1');
  }
  discharged = Builder(p.getContext()).getDenseBoolArrayAttr(values);
  return success();
}

ParseResult KGEN::parseBindParams(AsmParser &p, TypedAttr &generator,
                                  SmallVectorImpl<TypedAttr> &paramValues,
                                  DenseBoolArrayAttr &discharged,
                                  Type preParsedGeneratorType) {
  if (!preParsedGeneratorType &&
      parseColonTypeOrIndex(p, preParsedGeneratorType))
    return failure();

  if (parseParamValue(p, generator, preParsedGeneratorType))
    return failure();

  auto genType = sugarCast<GeneratorType>(preParsedGeneratorType);
  // Parse each operand, inferring its type from the signature type. Bound
  // parameters are allowed to refine the types of subsequent parameters, so
  // specialize the types as we go.
  ParameterEvaluator evaluator;
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);
  size_t numUnboundParams = 0;
  for (Type inputType : genType.getInputParamTypes()) {
    if (failed(p.parseOptionalComma()))
      break;
    Type remappedDeclType = evaluator.getReboundType(inputType);
    Type defaultValueType = minusOneAdjuster.replace(remappedDeclType);
    Type valueType;
    if (parseColonTypeOrDefault(p, valueType, defaultValueType) ||
        parseParamValue(p, paramValues.emplace_back(), valueType))
      return failure();
    TypedAttr value = paramValues.back();
    if (::isa<UnboundAttr>(value)) {
      auto residual = ParamIndexRefAttr::get(
          /*depth=*/-1, numUnboundParams++, defaultValueType);
      evaluator.appendIndexBinding(residual);
      continue;
    }
    evaluator.appendIndexBinding(value);
  }
  return parseDischargedBodyConstraints(p, discharged);
}

void KGEN::printBindParams(AsmPrinter &p, TypedAttr generator,
                           ArrayRef<TypedAttr> paramValues,
                           DenseBoolArrayAttr discharged) {
  printColonTypeParamValue(p, generator);
  auto genType = sugarCast<GeneratorType>(generator.getType());
  ParameterEvaluator evaluator;
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);
  size_t numUnboundParams = 0;
  for (auto [inputType, value] :
       llvm::zip(genType.getInputParamTypes(), paramValues)) {
    p << ", ";
    Type remappedDeclType = evaluator.getReboundType(inputType);
    Type defaultValueType = minusOneAdjuster.replace(remappedDeclType);
    if (::isa<UnboundAttr>(value)) {
      printColonTypeOrDefault(p, value.getType(), defaultValueType);
      printParamValue(p, value);
      auto residual = ParamIndexRefAttr::get(
          /*depth=*/-1, numUnboundParams++, defaultValueType);
      evaluator.appendIndexBinding(residual);
      continue;
    }
    // Be explicit about bound value types, because the binding type does not
    // necessarily have the same representation as the generator input
    // parameter type when both are inferred from outer scope.
    printColonTypeParamValue(p, value);
    evaluator.appendIndexBinding(value);
  }
  if (discharged && !discharged.empty()) {
    p << " | \"";
    for (bool value : discharged.asArrayRef())
      p << (value ? '1' : '0');
    p << '"';
  }
}

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringAttr name, bool isRef) {
  // If this will conflict with a reserved keyword then we need a '*' prefix and
  // double quotes.
  auto isSugaredType = [&] {
    return name.getContext()
        ->getLoadedDialect<KGENDialect>()
        ->typeParseFns.contains(name);
  };
  bool needsQuotes = !isLegalMLIRIdentifier(name) ||
                     (isRef && (succeeded(DType::getFromString(name)) ||
                                isMLIRBuiltinType(name) || isSugaredType()));
  if (needsQuotes)
    p << "*\"";
  printAsMojoStringLiteral(name, p.getStream());
  if (needsQuotes)
    p << '"';
}

ParseResult KGEN::parseParamNames(AsmParser &p,
                                  SmallVector<StringAttr> &names) {
  return p.parseCommaSeparatedList(
      [&] { return parseParamName(p, names.emplace_back()); });
}

void KGEN::printParamNames(AsmPrinter &p, ArrayRef<StringAttr> names,
                           bool isRef) {
  llvm::interleaveComma(
      names, p, [&](StringAttr name) { printParamName(p, name, isRef); });
}

/// Parse operator expression operands with operator-specific syntax.
static ParseResult parseOperatorOperands(AsmParser &p, uint32_t opcode,
                                         SmallVectorImpl<TypedAttr> &operands,
                                         Type type) {
  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    return p.parseCommaSeparatedList(
        [&] { return parseParamValue(p, operands.emplace_back(), type); });
  case (uint32_t)POC::TargetHasFeature:
  case (uint32_t)POC::TargetGetField:
    // Parse TargetHasFeature, and TargetGetField -- the first operand is a
    // TargetType, the second a StringType.
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        StringType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::GetSizeOf:
  case (uint32_t)POC::GetAlignOf:
    if (parseParamValue(p, operands.emplace_back(),
                        TypeType::get(p.getContext())) ||
        p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::Apply: {
    auto sigGen = dyn_cast_or_null<FuncTypeGeneratorType>(type);
    if (!sigGen)
      return p.emitError(p.getCurrentLocation(),
                         "expected a func type generator type for 'apply'");

    if (parseParamValue(p, operands.emplace_back(), sigGen))
      return failure();
    // Parse each operand, inferring its type from the signature type.
    // This adjuster is because the argument types in the signature have a
    // different depth than what the actual given arguments' types will be, see
    // STCHDDDOS.
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
    for (Type type : sigGen.getBody().getArguments())
      if (p.parseComma() ||
          parseParamValue(p, operands.emplace_back(), adjuster.replace(type)))
        return failure();
    return success();
  }
  case (uint32_t)POC::ApplyResultSlot: {
    auto sigGen = dyn_cast_or_null<FuncTypeGeneratorType>(type);
    if (!sigGen)
      return p.emitError(
          p.getCurrentLocation(),
          "expected a func type generator type for 'apply_result_slot'");
    FuncType sig = sigGen.getBody();

    if (parseParamValue(p, operands.emplace_back(), sigGen))
      return failure();
    if (sig.getNumArguments() < 1)
      return p.emitError(
          p.getCurrentLocation(),
          "'apply_result_slot' callee must have at least one result");
    // Parse each operand besides the result slot.
    // Adjust depth when the argument types in the signature have a
    // different depth than what the actual given arguments' types are, see
    // STCHDDDOS.
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
    auto argTypes = sig.getArguments().drop_back(sig.hasMemoryOnlyResult());
    for (Type type : argTypes)
      if (p.parseComma() ||
          parseParamValue(p, operands.emplace_back(), adjuster.replace(type)))
        return failure();
    return success();
  }
  case (uint32_t)POC::Cond:
    if (parseParamValue(p, operands.emplace_back(),
                        SIMDType::get(p.getContext(), 1, KGENDType::kBool)) ||
        p.parseComma() || parseParamValue(p, operands.emplace_back(), type) ||
        p.parseComma() || parseParamValue(p, operands.emplace_back(), type))
      return failure();
    return success();
  case (uint32_t)POC::GetEnv:
    return parseParamValue(p, operands.emplace_back(),
                           StringType::get(p.getContext()));
  case (uint32_t)POC::VariadicPtrMap:
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        IndexType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::DataToStr:
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseParamValue(p, operands.emplace_back(), ParamListType::get(type)))
      return failure();

    return success();

  case (uint32_t)POC::StringAddress:
    return parseParamValue(p, operands.emplace_back(),
                           StringType::get(type.getContext()));
  }
  llvm_unreachable("unknown operator");
}

static bool isI1OrSIMDOfBool(Type type) {
  return type.isSignlessInteger(1) || isSIMDOf<KGENDType::kBool>(type);
}

static uint32_t getOpcodeFromString(StringRef keyword) {
  // All the valid and builtin opcodes are legal.
  auto opcode = symbolizePOC(keyword);
  if (opcode.has_value())
    return (uint32_t)*opcode;

  if (keyword == "neg")
    return (uint32_t)POCAliases::NEG;
  if (keyword == "sub")
    return (uint32_t)POCAliases::SUB;
  if (keyword == "not")
    return (uint32_t)POCAliases::NOT;
  if (keyword == "ne")
    return (uint32_t)POCAliases::NE;
  if (keyword == "gt")
    return (uint32_t)POCAliases::GT;
  if (keyword == "ge")
    return (uint32_t)POCAliases::GE;

  return (uint32_t)POCAliases::kInvalid;
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
///
/// If 'disableTypeParser' is set, then the type parser is not used. This is
/// intended for cases where the type parser needs to recurse to handle general
/// cases.
ParseResult KGEN::parseParamValue(AsmParser &p, TypedAttr &value, Type type,
                                  bool disableTypeParser) {
  assert(type && "always have a contextual type");
  llvm::SMLoc loc = p.getCurrentLocation();

  // If the type provides a pretty parsing hook, use it.
  if (!disableTypeParser) {
    if (auto typeItf = dyn_cast<ParameterTypeInterface>(type)) {
      OptionalParseResult result = typeItf.parseValue(p, value);
      if (result.has_value())
        return *result;
    }
  }

  // If this is a '*'-prefixed double quoted string, then this is a simple
  // parameter reference.
  if (succeeded(p.parseOptionalStar())) {
    if (succeeded(p.parseOptionalLParen())) {
      // Try to parse '*()' as a singleton value, spelled after the unit type
      // '!kgen.struct<()>' it most often carries.
      if (succeeded(p.parseOptionalRParen())) {
        value = SingletonAttr::get(type);
        return success();
      }

      // Try to parse *(0,0) as an index reference.
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      value = ParamIndexRefAttr::get(depth, index, type);
      return success();
    }

    // Try to parse '*?' as an unknown value.
    if (succeeded(p.parseOptionalQuestion())) {
      value = UnknownAttr::get(type);
      return success();
    }

    std::string name;
    if (failed(p.parseString(&name)))
      return failure();
    value = ParamDeclRefAttr::get(name, type);
    return success();
  }

  // A '?' represents an unknown parameter.
  if (succeeded(p.parseOptionalQuestion())) {
    value = UnboundAttr::get(type);
    return success();
  }

  // Barewords / MLIR keywords are implicitly parameter declaration references
  // or the start of a expression in function form.
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Check to see if we're parsing a dtype name like 'f32'.
    if (isa<DTypeType>(type)) {
      auto dtype = KGENDType::getFromString(keyword);
      if (succeeded(dtype)) {
        value = DTypeConstantAttr::get(p.getContext(), *dtype);
        return success();
      }
    }

    // A bareword or string with no trailing `(` must be a parameter reference.
    if (failed(p.parseOptionalLParen())) {
      value = ParamDeclRefAttr::get(keyword, type);
      return success();
    }

    if (keyword == "from_builtin") {
      TypedAttr arg;
      if (parseColonTypeParamValue(p, arg) || p.parseRParen())
        return failure();
      // An optional trailing `: <type>` overrides the result type that would
      // otherwise be inferred from the arg's type.
      if (succeeded(p.parseOptionalColon())) {
        llvm::SMLoc typeLoc = p.getCurrentLocation();
        Type explicitType;
        if (parseKGENType(p, explicitType))
          return failure();
        auto simdType = dyn_cast<SIMDType>(explicitType);
        if (!simdType)
          return p.emitError(typeLoc,
                             "expected a SIMDType for 'from_builtin' result");
        value = CastFromBuiltinAttr::get(p.getContext(), arg, simdType);
      } else {
        value = CastFromBuiltinAttr::get(arg);
      }
      return success();
    }

    if (keyword == "to_builtin") {
      TypedAttr arg;
      if (parseColonTypeParamValue(p, arg) || p.parseRParen())
        return failure();
      // An optional trailing `: <type>` overrides the result type that would
      // otherwise be inferred from the arg's type.
      if (succeeded(p.parseOptionalColon())) {
        Type explicitType;
        if (parseKGENType(p, explicitType))
          return failure();
        value = CastToBuiltinAttr::get(p.getContext(), arg, explicitType);
      } else {
        value = CastToBuiltinAttr::get(arg);
      }
      return success();
    }

    // Otherwise it's a function expression.  If this has an explicit operand
    // type, parse it.
    Type operandType;
    OptionalParseResult typePresent = parseOptionalColonType(p, operandType);
    if (typePresent.has_value() && failed(typePresent.value()))
      return failure();

    // Decode the name as an operation code.
    auto opcode = getOpcodeFromString(keyword);

    // Handle other expressions with the same syntax as ParamOperatorAttr
    // TODO: Could turn this into a trait and push all this logic into the
    // attrs, which would also be nice for LIT attrs.
    if (opcode == (uint32_t)POCAliases::kInvalid) {
      if ((keyword == "upcast" || keyword == "downcast") && operandType) {
        TypedAttr operand;
        if (parseParamValue(p, operand, operandType))
          return failure();
        if (p.parseRParen())
          return failure();

        value = keyword == "upcast" ? UpcastAttr::get(type, operand)
                                    : DowncastAttr::get(type, operand);
        return success();
      }

      if (keyword == "bind_params" && operandType) {
        TypedAttr generator;
        SmallVector<TypedAttr> paramValues;
        DenseBoolArrayAttr discharged;
        if (parseBindParams(p, generator, paramValues, discharged, operandType))
          return failure();
        if (p.parseRParen())
          return failure();
        value = BindParamsAttr::get(p.getContext(), generator, paramValues,
                                    discharged);
        if (!isEqualCanon(value.getType(), type))
          return p.emitError(loc)
                 << "bind_params result type mismatch: expected " << type
                 << " but inferred " << value.getType();
        return success();
      }

      if (keyword == "conforms_to" && operandType) {
        TypedAttr operand;
        if (parseParamValue(p, operand, operandType))
          return failure();

        // The trait type is encoded as a `:<metatype> <value>` parameter value.
        TypedAttr traitType;
        if (p.parseComma() || parseColonTypeParamValue(p, traitType) ||
            p.parseRParen())
          return failure();

        value = TypeConformsToTraitAttr::get(operand, traitType);
        return success();
      }

      if (keyword == "identical" && operandType) {
        // All operands share the single leading `:<type>` prefix.
        SmallVector<TypedAttr> operands;
        if (p.parseCommaSeparatedList([&] {
              return parseParamValue(p, operands.emplace_back(), operandType);
            }) ||
            p.parseRParen())
          return failure();

        if (failed(ParamIdenticalAttr::verify(
                [&]() -> mlir::InFlightDiagnostic { return p.emitError(loc); },
                operands)))
          return failure();

        value = ParamIdenticalAttr::get(operands);
        return success();
      }

      if (keyword == "sugar_alias" || keyword == "sugar_builtin" ||
          keyword == "sugar_preserved") {
        TypedAttr sugared, expanded;
        if (parseParamValue(p, sugared, type) || p.parseComma() ||
            parseParamValue(p, expanded, type) || p.parseRParen())
          return failure();
        auto kind = keyword == "sugar_alias" ? SugarKind::Alias
                    : keyword == "sugar_builtin"
                        ? SugarKind::AlwaysInlineBuiltin
                        : SugarKind::Preserved;
        value = SugarAttr::get(sugared.getContext(), kind, /*memberName=*/{},
                               sugared, expanded);
        return success();
      }
      if (keyword == "sugar_member_alias") {
        Type baseType;
        StringAttr memberName;
        TypedAttr expanded;
        if (p.parseType(baseType) || p.parseComma() ||
            p.parseAttribute(memberName) || p.parseComma() ||
            parseParamValue(p, expanded, type) || p.parseRParen())
          return failure();
        value = SugarAttr::getMemberAlias(baseType, memberName, expanded);
        return success();
      }

      return p.emitError(loc, "unknown expression ") << keyword;
    }

    // Otherwise it is a ParamOperatorAttr.  Parse the operand list.
    SmallVector<TypedAttr> operands;

    // If there was no specified element type, then pick a default based on the
    // opcode in question.
    if (!operandType) {
      switch (opcode) {
      case (uint32_t)POC::EQ:
      case (uint32_t)POC::LT:
      case (uint32_t)POC::LE:
      case (uint32_t)POCAliases::NE:
      case (uint32_t)POCAliases::GE:
      case (uint32_t)POCAliases::GT:
        // Comparisons default to index type for their operand, since their
        // result is always `i1`.
        operandType = p.getBuilder().getIndexType();
        break;
      default:
        // Other operators default to the same operand type as the result type.
        operandType = type;
        break;
      }
    }

    // Parse the remaining operands.
    if (failed(p.parseOptionalRParen())) {
      if (parseOperatorOperands(p, opcode, operands, operandType) ||
          p.parseRParen())
        return failure();
    }

    // Desugar the negation operator from `neg(a)` to `mul(a, -1)`
    if (opcode == (uint32_t)POCAliases::NEG) {
      if (operands.size() != 1)
        return p.emitError(loc, "neg operator expects a single operand");
      operands.emplace_back(
          p.getBuilder().getIntegerAttr(operands[0].getType(), -1));
      opcode = (uint32_t)POC::Mul;
    }

    // Desugar the subtract operator from `sub(a, b)` to `add(a, mul(b, -1))`
    if (opcode == (uint32_t)POCAliases::SUB) {
      if (operands.size() != 2)
        return p.emitError(loc, "sub operator expects two operands");
      operands[1] = ParamOperatorAttr::getNeg(operands[1]);
      opcode = (uint32_t)POC::Add;
    }

    // If these are aliases for inverted i1 value, build the correct nodes.
    bool needsInvert = false;
    switch (opcode) {
    case (uint32_t)POCAliases::NE:
      opcode = (uint32_t)POC::EQ;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::GE:
      opcode = (uint32_t)POC::LT;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::GT:
      opcode = (uint32_t)POC::LE;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::NOT:
      if (operands.size() != 1 || !isI1OrSIMDOfBool(operands[0].getType()))
        return p.emitError(
            loc,
            "not operator returns a single operand of `i1` or `simd<bool>`");
      value = ParamOperatorAttr::getNot(operands[0]);
      return success();
    }

    // Okay, we parsed the operands, see if this is a valid expression.
    if (failed(ParamOperatorAttr::verify(
            [&]() -> mlir::InFlightDiagnostic { return p.emitError(loc); },
            (POC)opcode, operands, type)))
      return failure();
    // All is good, let's move!
    value =
        ParamOperatorAttr::get(type.getContext(), (POC)opcode, operands, type);

    // If we need to invert this, do so.
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);

    return success();
  }

  // Otherwise, we support other typed attributes as well, including dialect
  // define attributes, integers, strings, etc.
  return p.parseAttribute(value, type);
}

static void printOperatorOperands(AsmPrinter &p, POC opcode,
                                  ArrayRef<TypedAttr> operands) {
  // If the elements are not index type, print the type explicitly.
  if (llvm::is_contained({POC::EQ, POC::LT, POC::LE, POC::Rebind}, opcode))
    printColonTypeOrIndexPrefix(p, operands[0].getType());

  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    llvm::interleaveComma(
        operands, p, [&](TypedAttr operand) { printParamValue(p, operand); });
    break;

  case POC::Apply:
    // Print the signature operand with a type. Print all other operands without
    // types.
    printColonTypeOrIndexPrefix(p, operands.front().getType());
    printParamValue(p, operands.front());
    for (TypedAttr operand : operands.drop_front()) {
      p << ", ";
      printParamValue(p, operand);
    }
    break;

  case POC::ApplyResultSlot:
    // Print the signature operand with a type. Print all other operands without
    // types.
    printColonTypeOrIndexPrefix(p, operands.front().getType());
    printParamValue(p, operands.front());
    for (TypedAttr operand : operands.drop_front()) {
      p << ", ";
      printParamValue(p, operand);
    }
    break;

  case POC::Cond:
    printParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    p << ", ";
    printParamValue(p, operands[2]);
    break;

  case POC::PtrBitcast:
  case POC::LoadFromMem:
    printColonTypeParamValue(p, operands.front());
    break;
  case POC::VariadicPtrMap:
    // Type is of the list, but the index type doesn't need it.
    printColonTypeParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    break;
  case POC::VariadicPtrRemoveMap:
    // Include the type of the list.
    printColonTypeParamValue(p, operands[0]);
    break;
  case POC::DataToStr:
    p << ':';
    printKGENType(p, operands[0].getType());
    p << ' ';
    printParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    break;
  }
}

void KGEN::printAsMojoStringLiteral(StringRef name, raw_ostream &out) {
  for (unsigned char c : name) {
    switch (c) {
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\t':
      out << "\\t";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\a':
      out << "\\a";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\v':
      out << "\\v";
      break;
    default:
      if (llvm::isPrint(c) && c != '"')
        out << c;
      else
        out << '\\' << llvm::hexdigit(c >> 4) << llvm::hexdigit(c & 0x0F);
      break;
    }
  }
}

void KGEN::printParamValue(AsmPrinter &p, TypedAttr value, Type type) {
  // Use an alias for this attribute if available.
  if (succeeded(p.printAlias(value)))
    return;

  // If the attribute's type provides a pretty printing hook, try to use it.
  if (auto typeItf = dyn_cast<ParameterTypeInterface>(value.getType()))
    if (succeeded(typeItf.printValue(p, value)))
      return;

  if (isa<UnknownAttr>(value)) {
    p << "*?";
    return;
  }

  if (isa<SingletonAttr>(value)) {
    p << "*()";
    return;
  }

  if (isa<UnboundAttr>(value)) {
    p << '?';
    return;
  }

  if (auto bindParams = dyn_cast<BindParamsAttr>(value)) {
    p << "bind_params(";
    printBindParams(p, bindParams.getGenerator(), bindParams.getParamValues(),
                    bindParams.getDischarged());
    p << ')';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(value)) {
    bool isRef = isTypeExpr(value);
    if (auto type = dyn_cast<ParameterTypeInterface>(value.getType()))
      isRef |= type.isMetaType();
    printParamName(p, declRef.getName(), isRef);
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value)) {
    p << "*(" << indexRef.getDepth() << ',' << indexRef.getIndex() << ")";
    return;
  }

  // If this is a dtype constant with simple syntax, we can print it as a
  // keyword.
  if (auto dtypeConstant = dyn_cast<DTypeConstantAttr>(value)) {
    auto eltType = dtypeConstant.getDType();
    std::string stringRep = eltType.getAsString();
    // Don't allow things like complex<f64>.  We can extend this in the future
    // if there is a reason to of course.
    if (!StringRef(stringRep).contains('<')) {
      p << stringRep;
      return;
    }
  }

  if (auto from = dyn_cast<CastFromBuiltinAttr>(value)) {
    p << "from_builtin(";
    printColonTypeParamValue(p, from.getArg());
    p << ')';
    // If the actual result type differs from what would be inferred from the
    // arg's type, print it explicitly so the syntax round-trips.
    SIMDType inferredType =
        CastFromBuiltinAttr::getInferredResultType(from.getArg());
    if (inferredType != from.getType()) {
      p << " : ";
      printKGENType(p, from.getType());
    }
    return;
  }

  if (auto to = dyn_cast<CastToBuiltinAttr>(value)) {
    p << "to_builtin(";
    printColonTypeParamValue(p, to.getArg());
    p << ')';
    // If the actual result type differs from what would be inferred from the
    // arg's type, print it explicitly so the syntax round-trips.
    Type inferredType = CastToBuiltinAttr::getInferredResultType(to.getArg());
    if (inferredType != to.getType()) {
      p << " : ";
      printKGENType(p, to.getType());
    }
    return;
  }

  // Handle expressions.
  if (auto expr = dyn_cast<ParamOperatorAttr>(value)) {
    auto printExpr = [&](StringRef opcode, ArrayRef<TypedAttr> operands) {
      p << opcode << '(';
      printOperatorOperands(p, expr.getOpcode(), operands);
      p << ')';
    };

    auto isI1OrSIMDBoolConstant = [&](TypedAttr expr) {
      return isI1OrSIMDOfBool(expr.getType()) &&
             isa<IntegerAttr, SIMDAttr>(expr);
    };

    // If this is a inverted boolean sugar, handle it.
    if (expr.getOpcode() == POC::Xor && expr.getNumOperands() == 2 &&
        isI1OrSIMDBoolConstant(expr.getOperand(1))) {
      if (auto invertedExpr = dyn_cast<ParamOperatorAttr>(expr.getOperand(0))) {
        if (invertedExpr.getOpcode() == POC::EQ) {
          expr = invertedExpr;
          return printExpr("ne", expr.getOperands());
        }
        if (invertedExpr.getOpcode() == POC::LT) {
          expr = invertedExpr;
          return printExpr("ge", expr.getOperands());
        }
        if (invertedExpr.getOpcode() == POC::LE) {
          expr = invertedExpr;
          return printExpr("gt", expr.getOperands());
        }
      }

      // Otherwise, print as a generic "not".
      return printExpr("not", expr.getOperand(0));
    }

    return printExpr(stringifyEnum(expr.getOpcode()), expr.getOperands());
  }

  if (auto identical = dyn_cast<ParamIdenticalAttr>(value)) {
    // Operands all share one type, so print it once as a prefix.  Unlike `eq`
    // this always prints the prefix rather than defaulting to `index`, because
    // the parser needs it to type the operands.
    p << "identical(:";
    printKGENType(p, identical.getOperand(0).getType());
    p << ' ';
    llvm::interleaveComma(identical.getOperands(), p, [&](TypedAttr operand) {
      printParamValue(p, operand);
    });
    p << ')';
    return;
  }

  if (auto conformsTo = dyn_cast<TypeConformsToTraitAttr>(value)) {
    TypedAttr operand = conformsTo.getTypeValue();
    p << "conforms_to(:";
    printKGENType(p, operand.getType());
    p << ' ';
    printParamValue(p, operand);
    p << ", ";
    printColonTypeParamValue(p, conformsTo.getTraitType());
    p << ")";
    return;
  }

  // Handle other expressions with the same syntax as ParamOperatorAttr
  // TODO: Could turn this into a trait like ParameterTypeInterface and push all
  // this logic into the attrs, which would also be nice for LIT attrs.
  auto printCastAttr = [&](auto cast) {
    printKGENType(p, cast.getInputTypeValue().getType());
    p << ' ';
    printParamValue(p, cast.getInputTypeValue());
    p << ')';
  };

  if (auto upcast = dyn_cast<UpcastAttr>(value)) {
    p << "upcast(:";
    printCastAttr(upcast);
    return;
  }
  if (auto downcast = dyn_cast<DowncastAttr>(value)) {
    p << "downcast(:";
    printCastAttr(downcast);
    return;
  }

  // If this is an i1 integer attr, print it as zero or one; not true/false
  // keywords.  This simplifies the keyword processing logic.
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    if (intAttr.getType().isSignlessInteger(1)) {
      p << (intAttr.getValue().isZero() ? 0 : 1);
      return;
    }
  }

  if (auto sugar = dyn_cast<SugarAttr>(value)) {
    switch (sugar.getKind()) {
    case SugarKind::MemberAlias:
      p << "sugar_member_alias(";
      p.printType(sugar.getMemberAliasType());
      p << ", " << sugar.getMemberName() << ", ";
      printParamValue(p, sugar.getExpanded(), sugar.getType());
      p << ")";
      return;
    case SugarKind::Alias:
      p << "sugar_alias(";
      break;
    case SugarKind::AlwaysInlineBuiltin:
      p << "sugar_builtin(";
      break;
    case SugarKind::Preserved:
      p << "sugar_preserved(";
      break;
    }
    printParamValue(p, sugar.getSugared(), sugar.getType());
    p << ", ";
    printParamValue(p, sugar.getExpanded(), sugar.getType());
    p << ")";
    return;
  }

  p.printAttributeWithoutType(value);
}

void KGEN::printParamValue(AsmPrinter &p, Operation *op, TypedAttr value,
                           Type type) {
  printParamValue(p, value, type);
}

bool KGEN::isTypeExprType(Type type) {
  return isa<NonStructTypeType, TypeType>(type);
}

bool KGEN::isTypeExpr(TypedAttr attr) { return isTypeExprType(attr.getType()); }

std::optional<std::pair<TypedAttr, TypedAttr>>
KGEN::getIdentityProposition(TypedAttr prop) {
  // FIXME(MOCO-4577): a class of more than two is dropped rather than returning
  // every derived pair. That loses provability but stays sound.
  if (auto identical = sugarDynCast<ParamIdenticalAttr>(prop))
    if (identical.getNumOperands() == 2)
      return std::make_pair(identical.getOperand(0), identical.getOperand(1));

  auto op = sugarDynCast<ParamOperatorAttr>(prop);
  if (!op || op.getOpcode() != POC::EQ || op.getOperands().size() != 2)
    return std::nullopt;

  // A lane-wise `eq` answers identity only where the two coincide, which rules
  // out floats: IEEE equality holds between values that are not
  // interchangeable (`+0.0` and `-0.0`) and fails between a value and itself
  // (NaN). An unresolved dtype might still turn out to be a float.
  Type operandType = op.getOperand(0).getType();
  if (auto simdType = sugarDynCast<SIMDType>(operandType)) {
    std::optional<KGENDType> dtype = simdType.getResolvedDType();
    if (!dtype || !dtype->isIntLike())
      return std::nullopt;
  } else if (!operandType.isIntOrIndex()) {
    return std::nullopt;
  }
  return std::make_pair(op.getOperand(0), op.getOperand(1));
}

KGEN::EnvAttr KGEN::getModularEnvAttr(MLIRContext *ctx,
                                      CompilationContext *compileCtx) {
  NamedAttrList envAttrs;

#ifdef MODULAR_PRODUCTION
  envAttrs.set("MODULAR_PRODUCTION", IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_PRODUCTION

#if MODULAR_ENABLE_GPU_PROFILING
  envAttrs.set("MODULAR_ENABLE_GPU_PROFILING",
               IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_ENABLE_GPU_PROFILING

#if MODULAR_ENABLE_GPU_PROFILING_DETAILED
  envAttrs.set("MODULAR_ENABLE_GPU_PROFILING_DETAILED",
               IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_ENABLE_GPU_PROFILING_DETAILED

  envAttrs.set("BUILD_TYPE", StringAttr::get(STRINGIFY(BUILD_TYPE),
                                             KGEN::StringType::get(ctx)));
  envAttrs.set("MODULAR_ASYNCRT_MAX_PROFILING_LEVEL",
               IntegerAttr::get(IndexType::get(ctx),
                                MODULAR_ASYNCRT_MAX_PROFILING_LEVEL));

  if (compileCtx) {
    for (auto entry : compileCtx->mojoDefines) {
      auto k = entry.first;
#ifdef MODULAR_PRODUCTION
      // This is an end users release build. Pretend that the
      // `MODULAR_PRODUCTION` flag does not exist. This protects us from end
      // users trying to leak internal details.
      if (k == "MODULAR_PRODUCTION")
        continue;
#endif // MODULAR_PRODUCTION

      std::visit(
          [&](auto &&v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>) {
              // Subtle detail: a true value is represented as UnitAttr,
              // and false value is just not represented at all.
              if (BoolAttr::get(ctx, v) && v)
                envAttrs.set(k, UnitAttr::get(ctx));
            } else if constexpr (std::is_same_v<T, int>) {
              envAttrs.set(k, IntegerAttr::get(IndexType::get(ctx), v));
            } else if constexpr (std::is_same_v<T, std::string>) {
              envAttrs.set(k, StringAttr::get(v, KGEN::StringType::get(ctx)));
            } else {
              // NOTE: This should be a static_assert, but that breaks in torch
              // compile tests on some mac devices.
              assert("non-exhaustive visitor!");
            }
          },
          entry.second);
    }
  }

  return KGEN::EnvAttr::get(envAttrs.getDictionary(ctx));
}

KGEN::EnvAttr KGEN::getModuleEnvAttr(ModuleOp moduleOp) {
  if (moduleOp->hasAttrOfType<KGEN::EnvAttr>(KGEN::EnvAttr::getEnvAttrName()))
    return moduleOp->getAttrOfType<KGEN::EnvAttr>(
        KGEN::EnvAttr::getEnvAttrName());

  return EnvAttr::get(DictionaryAttr::get(moduleOp.getContext()));
}

void KGEN::extendWithModularEnvAttr(ModuleOp moduleOp,
                                    CompilationContext *compileCtx) {
  moduleOp->setAttr(KGEN::EnvAttr::getEnvAttrName(),
                    KGEN::getModularEnvAttr(moduleOp.getContext(), compileCtx)
                        .extend(getModuleEnvAttr(moduleOp)));
}

void KGEN::printIsMemoryOnly(AsmPrinter &p, TypedAttr isMemoryOnly) {
  if (auto boolAttr = dyn_cast<BoolAttr>(isMemoryOnly)) {
    if (boolAttr.getValue())
      p << " memoryOnly";
    return;
  }
  // Constraint proposition for conditional RegisterPassable.
  p << " memoryOnly(";
  printParamValue(p, isMemoryOnly, isMemoryOnly.getType());
  p << ")";
}

ParseResult KGEN::parseIsMemoryOnly(AsmParser &p, TypedAttr &isMemoryOnly) {
  if (succeeded(p.parseOptionalKeyword("memoryOnly"))) {
    // "memoryOnly" alone means unconditionally memory-only.
    // "memoryOnly(<expr>)" carries a constraint proposition.
    if (succeeded(p.parseOptionalLParen())) {
      Type i1Type = IntegerType::get(p.getContext(), 1);
      if (parseParamValue(p, isMemoryOnly, i1Type))
        return failure();
      if (p.parseRParen())
        return failure();
    } else {
      isMemoryOnly = BoolAttr::get(p.getContext(), true);
    }
    return success();
  }
  isMemoryOnly = BoolAttr::get(p.getContext(), false);
  return success();
}

void KGEN::printMinAlignment(AsmPrinter &p, TypedAttr minAlignment) {
  // Skip printing if minAlignment is the default (1) to keep IR clean.
  if (auto intAttr = dyn_cast<IntegerAttr>(minAlignment)) {
    if (intAttr.getInt() == 1)
      return;
    p << " align(" << intAttr.getInt() << ")";
  } else {
    // Future parametric support may produce other TypedAttr types.
    p << " align(" << minAlignment << ")";
  }
}

ParseResult KGEN::parseMinAlignment(AsmParser &p, TypedAttr &minAlignment) {
  if (succeeded(p.parseOptionalKeyword("align"))) {
    if (p.parseLParen())
      return failure();
    uint64_t value;
    if (p.parseInteger(value))
      return failure();
    if (p.parseRParen())
      return failure();
    minAlignment = IntegerAttr::get(IndexType::get(p.getContext()), value);
  } else {
    // Default alignment is 1 (no explicit alignment).
    minAlignment = IntegerAttr::get(IndexType::get(p.getContext()), 1);
  }
  return success();
}

ParseResult
KGEN::parseStructDefFields(AsmParser &p,
                           SmallVector<StructDefFieldAttr> &fields) {
  MLIRContext *ctx = p.getContext();
  return p.parseCommaSeparatedList([&]() {
    StringAttr name;
    TypedAttr typeValue;
    if (parseParamName(p, name) || p.parseColon() ||
        parseTypeParamValue(p, typeValue))
      return failure();
    fields.push_back(StructDefFieldAttr::get(ctx, name, typeValue));
    return mlir::success();
  });
}

void KGEN::printStructDefFields(AsmPrinter &p,
                                ArrayRef<StructDefFieldAttr> fields) {
  llvm::interleaveComma(fields, p, [&](StructDefFieldAttr field) {
    printParamName(p, field.getName());
    p << ": ";
    printTypeParamValue(p, field.getTypeValue());
  });
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

ParseResult KGEN::parseArgConvention(AsmParser &p, ArgConvention &convention) {
  StringRef effectStr;
  llvm::SMLoc loc = p.getCurrentLocation();
  // Parse an optional input convention specifier.
  convention = ArgConvention::ImmReg;
  if (succeeded(p.parseOptionalKeyword(&effectStr))) {
    if (std::optional<ArgConvention> conv = symbolizeArgConvention(effectStr)) {
      convention = *conv;
    } else {
      return p.emitError(loc, "expected a valid input convention");
    }
  }
  return success();
}

void KGEN::printArgConvention(AsmPrinter &p, ArgConvention convention) {
  if (convention != ArgConvention::ImmReg)
    p << ' ' << stringifyArgConvention(convention);
}

ParseResult KGEN::parseSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList) {
  OptionalParseResult result = parseOptionalSignatureValues(
      p, parseArg, values, effects, optionalResultList);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(), "expected '(' to begin signature");
}

void KGEN::printSignatureValues(AsmPrinter &p,
                                function_ref<void(unsigned)> printElt,
                                FunctionType functionType,
                                ArrayRef<ArgConvention> argConvs,
                                FnEffects fnEffects, bool optionalResultList) {
  p << '(';
  llvm::interleaveComma(llvm::seq<unsigned>(0, argConvs.size()), p, printElt);
  p << ')';

  // Print the function effects.
  impl::FnEffects effects = fnEffects.getImpl();
  if (effects != impl::FnEffects::None)
    p << ' ' << impl::stringifyFnEffects(effects);

  if (optionalResultList)
    p.printOptionalArrowTypeList(functionType.getResults());
  else
    p.printArrowTypeList(functionType.getResults());
}

ParseResult KGEN::parseFunctionFuncTypeGenerator(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, FuncTypeGeneratorType &signature,
    ParamDeclParseHookTy parseDeclElt) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<ArgConvention> argConventions;
  FnEffects effects;
  if (parseOptionalParameterSpec(p, inputParams, resultParams, parseDeclElt))
    return failure();

  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    // Parse the argument type and its input convention.
    OpAsmParser::Argument &arg = args.emplace_back();
    OptionalParseResult result =
        p.parseOptionalArgument(arg, /*allowType=*/true);

    // An SSA name is present and resulted in a parsing error.
    if (result.has_value() && failed(*result))
      return failure();

    // An SSA name is not present, try parsing just the type.
    if (!result.has_value()) {
      // Failed to parse the type as well.
      if (p.parseType(arg.type))
        return failure();

      // Without an SSA name, the location information will not be set for
      // the argument, use the current parser location.
      arg.ssaName.location = p.getCurrentLocation();
    }

    if (parseArgConvention(p, argConventions.emplace_back()))
      return failure();

    argTypes.push_back(arg.type);
    return success();
  };

  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  signature = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      inputParams, functionType, argConventions, effects, {}, {},
      [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printFunctionFuncTypeGenerator(OpAsmPrinter &p, Region *region,
                                          ArrayRef<ParamDeclAttr> inputParams,
                                          ArrayRef<ParamDeclAttr> resultParams,
                                          FunctionType functionType,
                                          FuncTypeGeneratorType signature,
                                          ParamDeclPrintHookTy printInputElt,
                                          ParamDeclPrintHookTy printResultElt) {
  // Print the function arguments.
  FuncType sigBase = signature.getBody();
  auto printElt = [&](unsigned i) {
    if (!region)
      p << functionType.getInput(i);
    else
      p.printRegionArgument(region->getArgument(i));

    printArgConvention(p, sigBase.getArgConvention(i));
  };

  printOptionalParameterSpec(p, inputParams, resultParams, printInputElt,
                             printResultElt);
  printSignatureValues(p, printElt, functionType, sigBase.getArgConventions(),
                       sigBase.getFnEffects(),
                       /*optionalResultList=*/true);
}

ParseResult KGEN::parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    function_ref<ParseResult(SmallVectorImpl<Type> &)> parseInputTy) {
  if (failed(p.parseOptionalLess()) || succeeded(p.parseOptionalGreater()))
    return success();

  auto defaultParseInputTy = [&](SmallVectorImpl<Type> &inputs) {
    return parseKGENType(p, inputs.emplace_back());
  };
  if (!parseInputTy)
    parseInputTy = defaultParseInputTy;

  // Parse the input parameter types.
  auto parseIn = [&]() { return parseInputTy(inputParamTypes); };
  if (p.parseCommaSeparatedList(parseIn))
    return failure();

  if (p.parseGreater())
    return failure();
  return success();
}

void KGEN::printOptionalParamSignature(AsmPrinter &p,
                                       ArrayRef<Type> inputParamTypes,
                                       function_ref<void(Type)> printInputTy) {
  if (inputParamTypes.empty())
    return;

  auto defaultPrintInputTy = [&](Type type) { printKGENType(p, type); };
  if (!printInputTy)
    printInputTy = defaultPrintInputTy;

  p << '<';
  llvm::interleaveComma(inputParamTypes, p, printInputTy);
  p << '>';
}

ParseResult KGEN::parseFuncType(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalNewKGENSignature(p, signature);
  if (result.has_value())
    return *result;
  result = p.parseOptionalType(signature);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(),
                       "expected '<' or '(' to begin a signature");
  if (failed(*result))
    return failure();
  if (!isa<FuncType>(signature))
    return p.emitError(p.getCurrentLocation(), "expected a signature type");
  return success();
}

void KGEN::printFuncType(AsmPrinter &p, FuncType signature) {
  // If the signature has metadata, ask its dialect to print the signature.
  if (FnMetadataAttrInterface metadata = signature.getMetadata()) {
    metadata.printFuncType(p, signature);
    return;
  }

  auto printElt = [&](unsigned i) {
    p << signature.getArgument(i);
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, signature.getValues(),
                       signature.getArgConventions(), signature.getFnEffects(),
                       /*optionalResultList=*/false);
}

ParseResult KGEN::parseKGENFuncTypeGenerator(AsmParser &p,
                                             FunctionType &functionType,
                                             FuncTypeGeneratorType &generator) {
  Type type;
  if (parseGenerator(p, type))
    return failure();
  generator = dyn_cast<FuncTypeGeneratorType>(type);
  if (!generator)
    return failure();
  functionType = generator.getBody().getValues();
  return success();
}

ParseResult KGEN::parseGenerator(AsmParser &p, Type &generator) {
  // Try parsing as a plain KGEN generator first (no metadata);
  OptionalParseResult result = parseOptionalKGENGenerator(p, generator);
  if (result.has_value())
    return *result;

  result = p.parseOptionalType(generator);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(),
                       "expected '<' to begin a generator");
  if (failed(*result))
    return failure();
  if (!isa<GeneratorType>(generator))
    return p.emitError(p.getCurrentLocation(), "expected a generator type");
  return success();
}

void KGEN::printGenerator(AsmPrinter &p, GeneratorType generator) {
  if (PogListAttr metadata = generator.getParamListAttrs()) {
    metadata.printGenerator(p, generator);
    return;
  }

  // For maximum textual IR back-compat, skip printing the empty angle brackets
  // for func type generators. We should remove this sugar after the migration.
  if (!isa<FuncType>(generator.getBody()) ||
      !generator.getInputParamTypes().empty()) {
    p << '<';
    printParamTypes(p, generator.getInputParamTypes());
    p << '>';
  }
  printParamType(p, generator.getBody());
}

void KGEN::printSignatureValues(AsmPrinter &p, FunctionType functionType,
                                FuncTypeGeneratorType sigGen) {
  // If the signature has metadata, ask its dialect to print the signature.
  if (PogListAttr metadata = sigGen.getParamListAttrs()) {
    metadata.printGenerator(p, sigGen);
    return;
  }

  FuncType signature = sigGen.getBody();
  auto printElt = [&](unsigned i) {
    p << functionType.getInput(i);
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, functionType, signature.getArgConventions(),
                       signature.getFnEffects(),
                       /*optionalResultList=*/false);
}

ParseResult KGEN::parseOptionalDecorators(AsmParser &p,
                                          DecoratorsAttr &decorators) {
  SmallVector<TypedAttr> decoVals;
  if (succeeded(p.parseOptionalKeyword("decorators"))) {
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&] {
          return parseColonTypeParamValue(p, decoVals.emplace_back());
        }))
      return failure();
  }
  decorators = DecoratorsAttr::get(p.getContext(), decoVals);
  return success();
}

void KGEN::printOptionalDecorators(OpAsmPrinter &p, Operation *op,
                                   ArrayRef<TypedAttr> decorators) {
  if (decorators.empty())
    return;
  p.printNewline();
  p << "  decorators <";
  llvm::interleaveComma(decorators, p, [&](TypedAttr decorator) {
    if (decorators.size() > 1) {
      p.printNewline();
      p << "    ";
    }
    printColonTypeParamValue(p, decorator);
  });
  p << ">";
}

/// Parse the always_inline related keywords if present.
ParseResult KGEN::parseOptionalInline(OpAsmParser &parser,
                                      InlineLevelAttr &attr) {
  // Handle always_inline.
  InlineLevel inlineLevel;
  if (succeeded(parser.parseOptionalKeyword("always_inline")))
    inlineLevel = InlineLevel::Always;
  else if (succeeded(parser.parseOptionalKeyword("always_inline_no_debug")))
    inlineLevel = InlineLevel::AlwaysNoDebug;
  else if (succeeded(parser.parseOptionalKeyword("always_inline_builtin")))
    inlineLevel = InlineLevel::AlwaysBuiltin;
  else if (succeeded(parser.parseOptionalKeyword("no_inline")))
    inlineLevel = InlineLevel::Never;
  else
    inlineLevel = InlineLevel::Automatic;
  attr = InlineLevelAttr::get(parser.getContext(), inlineLevel);
  return success();
}

void KGEN::printOptionalInline(AsmPrinter &p, InlineLevel level) {
  switch (level) {
  case InlineLevel::Automatic:
    break;
  case InlineLevel::Always:
    p << " always_inline";
    break;
  case InlineLevel::AlwaysNoDebug:
    p << " always_inline_no_debug";
    break;
  case InlineLevel::AlwaysBuiltin:
    p << " always_inline_builtin";
    break;
  case InlineLevel::Never:
    p << " no_inline";
    break;
  }
}

ParseResult KGEN::parseSymbolExport(AsmParser &p, ExportKindAttr &exportKind) {
  ExportKind value = ExportKind::NotExported;
  if (succeeded(p.parseOptionalKeyword("export")))
    value = ExportKind::Exported;
  exportKind = ExportKindAttr::get(p.getContext(), value);
  return success();
}

void KGEN::printSymbolExport(AsmPrinter &p, Operation *op,
                             ExportKindAttr exportKind) {
  if (exportKind.getValue() != ExportKind::NotExported)
    p << " export";
}

ParseResult KGEN::parseParameterValues(AsmParser &p,
                                       ParameterExprArrayAttr &values) {
  SmallVector<TypedAttr> elts;
  if (parseParameterValues(p, elts))
    return failure();
  values = ParameterExprArrayAttr::get(p.getContext(), elts);
  return success();
}

ParseResult KGEN::parseParameterValues(AsmParser &p,
                                       SmallVectorImpl<TypedAttr> &values) {
  return p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
        TypedAttr value;
        if (parseParamValueDefaultingToIndex(p, value))
          return failure();
        values.push_back(value);
        return success();
      });
}

void KGEN::printParameterValues(OpAsmPrinter &p, Operation *op,
                                ParameterExprArrayAttr values) {
  printParameterValues(p, values);
}

void KGEN::printParameterValues(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  if (values.empty())
    return;
  p << '<';
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    auto valType = value.getType();
    printColonTypeOrIndexPrefix(p, valType);
    printParamValue(p, value);
  });
  p << '>';
}

ParseResult KGEN::parseParametricCallee(OpAsmParser &p, TypedAttr &callee) {
  Type type;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseLSquare() || parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, callee, type) || p.parseRSquare())
    return failure();

  if (!isa<ParamType, FuncTypeGeneratorType>(callee.getType()))
    return p.emitError(
               loc,
               "callee parameter type must be a func type generator type. Got ")
           << callee.getType();
  return success();
}

void KGEN::printParametricCallee(OpAsmPrinter &p, Operation *,
                                 TypedAttr callee) {
  p << "[";
  printKGENType(p, callee.getType());
  p << ": ";
  printParamValue(p, callee);
  p << "]";
}

void KGEN::printEmissionKind(AsmPrinter &p, TypedAttr emissionKind) {
  // '=' is used to disambiguate the string form.
  if (auto emitAsAttr = dyn_cast<EmitAsAttr>(emissionKind))
    p << '=' << stringifyEmitAs(emitAsAttr.getValue());
  else
    printParamValue(p, emissionKind);
}

ParseResult KGEN::parseEmissionKind(AsmParser &p, TypedAttr &emissionKind) {
  if (succeeded(p.parseOptionalEqual())) {
    StringRef emissionKindStr;
    if (p.parseKeyword(&emissionKindStr))
      return failure();
    std::optional<EmitAs> kind = symbolizeEmitAs(emissionKindStr);
    if (!kind) {
      return p.emitError(p.getCurrentLocation(),
                         "the immediate emission kind must be either "
                         "'=llvm', '=asm', '=llvm-opt', or '=object'");
    }
    emissionKind = cast<TypedAttr>(EmitAsAttr::get(p.getContext(), *kind));
  } else if (parseParamValue(p, emissionKind, p.getBuilder().getIndexType())) {
    return failure();
  }
  return success();
}

/// Compare a range of values from an "originator" to a corresponding range of
/// values from a "target".  If the two mismatch, emit an error that tries to
/// explain the issue in a nice way.
template <typename TargetRange, typename OriginatorRange>
static ParseResult verifyMatchingLists(
    const OriginatorRange &originatorRange, const TargetRange &targetRange,
    StringRef originatorName, Location originatorLoc, StringRef targetName,
    Location targetLoc, StringRef itemName, StringRef propertyName) {
  // Check that the ranges have the same size.  If not, diagnose this.
  size_t numOriginator =
      std::distance(originatorRange.begin(), originatorRange.end());
  size_t numTarget = std::distance(targetRange.begin(), targetRange.end());
  if (numOriginator != numTarget) {
    auto diag = emitError(originatorLoc, originatorName)
                << " has " << numOriginator << " " << itemName
                << (numOriginator != 1 ? "s" : "") << " but @" << targetName
                << " expects " << numTarget;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << "@" << targetName << " declared here";
    return failure();
  }

  // If they have the same sizes, diagnose any mismatches between their
  // elements.

  // NOTE: llvm::zip doesn't work with LLVM mapped iterators.
  auto targetIt = targetRange.begin();
  auto originatorIt = originatorRange.begin();
  for (size_t itemNum = 0; itemNum != numTarget; ++itemNum) {
    auto targetVal = *targetIt++;
    auto originatorVal = *originatorIt++;
    // The types or attributes must equal, but can differ on sugar.
    if (originatorVal == targetVal)
      continue;
    if constexpr (std::is_base_of_v<TypedAttr, decltype(originatorVal)> ||
                  std::is_base_of_v<Type, decltype(originatorVal)>) {
      if (isEqualCanon(originatorVal, targetVal))
        continue;
    }

    auto diag = emitError(originatorLoc, originatorName)
                << ' ' << itemName << " #" << itemNum << " has " << propertyName
                << ' ' << originatorVal << " but @" << targetName
                << " expected " << propertyName << ' ' << targetVal;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << "@" << targetName << " declared here";
    return failure();
  }

  return success();
}

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult KGEN::verifyDeclSignaturesMatch(
    StringRef lhsName, FuncTypeGeneratorType lhsSigGen, Location lhsLoc,
    StringRef rhsName, FuncTypeGeneratorType rhsSigGen, Location rhsLoc) {
  VerboseCompilerTimeTraceScope traceScope("verifyDeclSignaturesMatch");

  if (failed(verifyMatchingLists(
          lhsSigGen.getInputParamTypes(), rhsSigGen.getInputParamTypes(),
          lhsName, lhsLoc, rhsName, rhsLoc, "input parameter", "type"))) {
    return failure();
  }
  return verifyFuncTypesMatch(lhsName, lhsSigGen.getBody(), lhsLoc, rhsName,
                              rhsSigGen.getBody(), rhsLoc);
}

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult KGEN::verifyFuncTypesMatch(StringRef lhsName, FuncType lhsSig,
                                         Location lhsLoc, StringRef rhsName,
                                         FuncType rhsSig, Location rhsLoc) {
  VerboseCompilerTimeTraceScope traceScope("verifyFuncTypesMatch");

  FunctionType lhsType = lhsSig.getValues();
  FunctionType rhsType = rhsSig.getValues();

  /// Verify that a list of parameter declarations from a generator or func
  /// matches those of an interface.  This produces an error diagnostic and
  /// returns failure when a problem is detected, or returns true if
  /// everything is ok.
  if (verifyMatchingLists(lhsType.getInputs(), rhsType.getInputs(), lhsName,
                          lhsLoc, rhsName, rhsLoc, "argument", "type") ||
      verifyMatchingLists(lhsType.getResults(), rhsType.getResults(), lhsName,
                          lhsLoc, rhsName, rhsLoc, "result", "type") ||
      verifyMatchingLists(lhsSig.getArgConventions(),
                          rhsSig.getArgConventions(), lhsName, lhsLoc, rhsName,
                          rhsLoc, "argument", "convention"))
    return failure();

  if (lhsSig.getFnEffects() != rhsSig.getFnEffects()) {
    auto diag = emitError(lhsLoc, lhsName)
                << " function effects are " << lhsSig.getFnEffects() << " but @"
                << rhsName << " expected " << rhsSig.getFnEffects();
    if (lhsLoc != rhsLoc)
      diag.attachNote(rhsLoc) << rhsName << " declared here";
    return failure();
  }

  FnMetadataAttrInterface lhsMetadata = lhsSig.getMetadata();
  FnMetadataAttrInterface rhsMetadata = rhsSig.getMetadata();
  if (lhsMetadata != rhsMetadata) {
    // Metadata itself is not a parameter, so canonicalization will not modify
    // the top level attribute kind.
    auto lhsCanonMetadata =
        cast<FnMetadataAttrInterface>(getCanonicalAttr(lhsMetadata));
    auto rhsCanonMetadata =
        cast<FnMetadataAttrInterface>(getCanonicalAttr(rhsMetadata));
    if (lhsCanonMetadata != rhsCanonMetadata &&
        !lhsCanonMetadata.equals(rhsCanonMetadata)) {
      auto diag = emitError(lhsLoc, lhsName)
                  << " metadata is " << lhsSig.getMetadata() << " but @"
                  << rhsName << " expected " << rhsSig.getMetadata();
      if (lhsLoc != rhsLoc)
        diag.attachNote(rhsLoc) << rhsName << " declared here";
      return failure();
    }
  }

  return success();
}

LogicalResult
KGEN::verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                            ArrayRef<TypedAttr> paramValues,
                            Location originatorLoc, StringRef targetName,
                            ArrayRef<ParamDeclAttr> decls, Location targetLoc) {
  using llvm::map_range;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  return verifyMatchingLists(
      map_range(paramValues, getType), map_range(decls, getType),
      originatorName, originatorLoc, targetName, targetLoc, paramKind, "type");
}

LogicalResult KGEN::verifyCallOperands(Operation *op, ValueRange args,
                                       FuncType callee, bool ignoreByRef) {
  unsigned numByRef = ignoreByRef * callee.getNumAsyncReturnSlots();
  if (args.size() != callee.getNumArguments() - numByRef) {
    return op->emitOpError("callee expected ")
           << callee.getNumArguments() << " arguments but operation only has "
           << args.size();
  }
  for (auto [i, arg, type] :
       llvm::enumerate(args, callee.getArguments().drop_back(numByRef))) {
    if (arg.getType() != type) {
      return op->emitOpError("callee argument #")
             << i << " expected type " << type
             << " but operation argument has type " << arg.getType();
    }
  }
  return success();
}

LogicalResult KGEN::verifyCallResults(Operation *op, ValueRange results,
                                      FuncType callee) {
  if (results.size() != callee.getNumResults()) {
    return op->emitOpError("callee expected ")
           << callee.getNumArguments() << " results but operation only has "
           << results.size();
  }
  for (auto [i, res, type] : llvm::enumerate(results, callee.getResults())) {
    if (res.getType() != type) {
      return op->emitOpError("callee result #")
             << i << " expected type " << type
             << " but operation result has type " << res.getType();
    }
  }
  return success();
}

ExportMap KGEN::getExportedSymbols(ModuleOp module) {
  ExportMap exportedSymbols;
  for (auto op : module.getOps<ExportInterface>()) {
    if (op.isExported())
      exportedSymbols.insert({op.getSymNameAttr(), op.getExportKind()});
  }
  return exportedSymbols;
}

/// Return if the given decorator matches an annotation, whose scopes are split
/// into the given parts.
static bool isDecorator(TypedAttr decorator,
                        ArrayRef<StringRef> annotationParts) {
  if (auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator))
    decorator = apply.getOperand(0);

  auto sym = dyn_cast<KGEN::SymbolConstantAttr>(decorator);
  if (!sym)
    return false;
  SymbolRefAttr symRef = sym.getSymbol();
  ArrayRef<FlatSymbolRefAttr> nestedRefs = symRef.getNestedReferences();

  // Check the root reference.
  if (symRef.getRootReference() != annotationParts.front() ||
      nestedRefs.size() != annotationParts.size() - 1)
    return false;
  // Check the middle references.
  for (int i = 0, e = annotationParts.size() - 2; i < e; ++i)
    if (nestedRefs[i].getValue() != annotationParts[i + 1])
      return false;
  // Check the leaf reference.
  return nestedRefs.back().getValue().starts_with(annotationParts.back());
}

bool KGEN::hasDecorator(ArrayRef<TypedAttr> decorators, StringRef annotation) {
  SmallVector<StringRef> parts;
  annotation.split(parts, "::");
  return llvm::any_of(decorators, [&](TypedAttr decorator) {
    return isDecorator(decorator, parts);
  });
}

bool KGEN::hasAnyDecorator(ArrayRef<TypedAttr> decorators,
                           ArrayRef<StringLiteral> annotations) {
  return llvm::any_of(annotations, [&](const StringLiteral &annot) {
    return hasDecorator(decorators, annot);
  });
}

ParseResult KGEN::parseRegionWithArgs(OpAsmParser &p, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (p.parseArgumentList(args, AsmParser::Delimiter::OptionalParen,
                          /*allowType=*/true) ||
      p.parseRegion(region, args))
    return failure();
  return success();
}

void KGEN::printRegionWithArgs(OpAsmPrinter &p, Operation *op, Region &region) {
  if (!region.getArguments().empty()) {
    p << '(';
    llvm::interleaveComma(region.getArguments(), p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ") ";
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

std::string KGEN::printSimpleParamAttrValues(
    ArrayRef<ParamDeclAttr> params, ArrayRef<TypedAttr> values,
    CompilationOptions::ErrorVerboseLevel verboseLevel) {
  SmallVector<std::string> result;
  for (auto [param, value] : llvm::zip(params, values)) {
    llvm::TypeSwitch<TypedAttr>(value)
        .Case<StringAttr>([&](auto &attr) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << param.getName() << ": " << attr.getValue();
          result.push_back(str);
        })
        .Case<BoolAttr>([&](auto &attr) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << param.getName() << ": " << attr;
          result.push_back(str);
        })
        .Case<IntegerAttr>([&](auto &attr) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << param.getName() << ": " << attr.getValue();
          result.push_back(str);
        })
        .Case<FloatAttr>([&](auto &attr) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << param.getName() << ": " << attr.getValue();
          result.push_back(str);
        })
        .Case<KGEN::SIMDAttr>([&](auto &attr) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << param.getName() << ": ";
          KGEN::printDTypeValues(os, attr.getValues(),
                                 *attr.getType().getResolvedDType());
          result.push_back(str);
        })
        .Default([&](auto &attr) {
          if (verboseLevel == CompilationOptions::kAllParams) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << param.getName() << ": " << attr;
            result.push_back(str);
          } else {
            result.push_back("...");
          }
        });
  }

  std::string str;
  if (verboseLevel != CompilationOptions::kNoParams && !result.empty()) {
    llvm::raw_string_ostream os(str);
    os << "(";
    llvm::interleaveComma(result, os);
    os << ")";
  }

  return str;
}

//===----------------------------------------------------------------------===//
// CastFromBuiltin / CastToBuiltin folding helpers
//===----------------------------------------------------------------------===//

LogicalResult KGEN::verifyConversionCast(
    function_ref<InFlightDiagnostic(StringRef)> emitError, SIMDType simd,
    Type builtinType, bool fromSimd) {
  // Verify the SIMD size matches the vector size and the dtypes match.
  auto size = simd.getResolvedSize();
  if (size && *size == 1) {
    // Scalar case
    auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
    if (dtype && !dtype.isConvertibleTo(builtinType))
      return emitError("cannot convert ")
             << (fromSimd ? "from" : "to") << " scalar dtype "
             << dtype.getDType().getAsString() << (fromSimd ? " to " : " from ")
             << builtinType;
    return success();
  }

  auto vector = dyn_cast<VectorType>(builtinType);
  if (!vector || vector.getRank() != 1 || vector.isScalable())
    return emitError("expected a rank 1 non-scalable vector");

  if (size && *size != vector.getShape().front())
    return emitError("expected vector<") << *size << "xT>";

  if (auto dtype = dyn_cast<DTypeConstantAttr>(simd.getDType());
      dtype && !dtype.isConvertibleTo(vector.getElementType()))
    return emitError("cannot convert ")
           << (fromSimd ? "from" : "to") << " SIMD dtype "
           << dtype.getDType().getAsString() << (fromSimd ? " to" : " from")
           << " vector element " << vector.getElementType();
  return success();
}

//===----------------------------------------------------------------------===//
// SIMD Utilities
//===----------------------------------------------------------------------===//

/// Convert a SIMD attribute to a vector-typed attribute.
template <typename AttrT, typename TransformFn>
static ArrayElementsAttr convertSIMDToVectorAttr(SIMDAttr simd, VectorType type,
                                                 TransformFn fn) {
  SmallVector<decltype(fn(std::declval<DTypeValue>()))> values;
  for (const DTypeValue &value : simd.getValues())
    values.push_back(fn(value));
  return AttrT::get(type, values);
}

OpFoldResult KGEN::foldCastToBuiltin(TypedAttr input, Type resultType) {
  // Look through sugar to fold.
  input = SugarAttr::strip(input);

  if (auto cast = sugarDynCastIfPresent<CastFromBuiltinAttr>(input))
    if (cast.getArg().getType() == resultType)
      return cast.getArg();

  auto simd = sugarDynCastIfPresent<SIMDAttr>(input);
  if (!simd)
    return {};
  // Conversion to a 1D vector type.
  std::optional<KGENDType> dtype = simd.getType().getResolvedDType();
  if (!dtype)
    return {};

  if (auto vector = dyn_cast<VectorType>(resultType)) {
    if (dtype->isBool())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector,
          [](DTypeValue simd) { return APInt(1, simd.getBoolVal()); });
    if (dtype->isIndex() || dtype->isUIndex())
      return convertSIMDToVectorAttr<IndexArrayElementsAttr>(
          simd, vector, [](DTypeValue simd) { return simd.getIndexVal(); });
    if (dtype->isInt())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector, [](DTypeValue simd) { return simd.getIntVal(); });
    assert(dtype->isFloat() && "unexpected dtype");
    return convertSIMDToVectorAttr<FloatArrayElementsAttr>(
        simd, vector, [](DTypeValue simd) { return simd.getFloatVal(); });
  }

  assert(simd.getValues().size() == 1 && "expected a scalar constant");
  const DTypeValue &value = simd.getValues().front();

  // Convert to a scalar attribute.
  Builder b(simd.getContext());
  if (dtype->isBool())
    return b.getBoolAttr(value.getBoolVal());
  if (dtype->isIndex() || dtype->isUIndex())
    return b.getIndexAttr(value.getIndexVal());
  if (dtype->isInt())
    return b.getIntegerAttr(cast<IntegerType>(resultType), value.getIntVal());
  assert(dtype->isFloat() && "unexpected dtype");
  return b.getFloatAttr(cast<FloatType>(resultType), value.getFloatVal());
}

OpFoldResult KGEN::foldCastFromBuiltin(TypedAttr val, SIMDType resultType) {
  // Look through sugar to fold.
  val = SugarAttr::strip(val);
  if (auto cast = sugarDynCastIfPresent<CastToBuiltinAttr>(val))
    if (cast.getArg().getType() == resultType)
      return cast.getArg();

  // Ensure the incoming value is an expected constant kind.
  if (!isa<IntArrayElementsAttr, FloatArrayElementsAttr, IndexArrayElementsAttr,
           IntegerAttr, FloatAttr>(val))
    return {};

  // Conversion from vector constant.
  std::optional<KGENDType> dtype = resultType.getResolvedDType();
  if (!dtype)
    return {};
  if (auto vector = dyn_cast<VectorType>(val.getType())) {
    SmallVector<DTypeValue> values;
    if (dtype->isBool())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(!value.isZero(), *dtype);
    else if (dtype->isIndex() || dtype->isUIndex())
      for (int64_t value : cast<IndexArrayElementsAttr>(val))
        values.emplace_back(value, *dtype);
    else if (dtype->isInt())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    else
      for (APFloat value : cast<FloatArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    return SIMDAttr::get(values, resultType);
  }

  // Handle scalar constants.
  if (dtype->isBool())
    return SIMDAttr::get({cast<BoolAttr>(val).getValue(), *dtype}, resultType);
  if (dtype->isIndex() || dtype->isUIndex())
    return SIMDAttr::get({cast<IntegerAttr>(val).getInt(), *dtype}, resultType);
  if (dtype->isInt())
    return SIMDAttr::get({cast<IntegerAttr>(val).getValue(), *dtype},
                         resultType);
  assert(dtype->isFloat() && "unexpected dtype");
  return SIMDAttr::get({cast<FloatAttr>(val).getValue(), *dtype}, resultType);
}

OpFoldResult KGEN::foldSIMDSplat(Value scalarVal, Attribute scalarAttr,
                                 SIMDType resultType) {
  std::optional<int64_t> size = resultType.getResolvedSize();

  if (size == 1)
    return scalarAttr ? OpFoldResult(scalarAttr) : scalarVal;

  auto scalarSIMD = sugarDynCastIfPresent<SIMDAttr>(scalarAttr);
  if (!size || size <= 0 || !scalarSIMD)
    return {};
  SmallVector<DTypeValue> values(*size, scalarSIMD.getValues().front());
  return SIMDAttr::get(values, resultType);
}

SIMDType KGEN::getEquivalentSIMDType(Type type) {
  auto [dtype, size] = KGENDType::getEquivalentDType(type);
  if (dtype.isInvalid())
    return {};

  return SIMDType::get(size.value_or(1),
                       DTypeConstantAttr::get(type.getContext(), dtype));
}

TypedAttr KGEN::splatBuiltinToSIMD(TypedAttr builtinScalarVal,
                                   TypedAttr simdSize) {
  SIMDType simdScalarTp = getEquivalentSIMDType(builtinScalarVal.getType());
  if (!simdScalarTp)
    return {};

  auto simdScalarVal = CastFromBuiltinAttr::get(builtinScalarVal, simdScalarTp);
  return SIMDSplatAttr::get(simdSize.getContext(), simdScalarVal,
                            SIMDType::get(simdSize, simdScalarTp.getDType()));
}

template <typename T>
TypedAttr splatLiteralToSIMDImpl(T literal, SIMDType target) {
  std::optional<KGENDType> dtype = target.getResolvedDType();
  if (!dtype)
    return {};

  auto scalarType = SIMDType::get(target.getContext(), 1, *dtype);
  SIMDAttr scalar;
  if constexpr (std::is_same_v<T, double>) {
    assert(dtype->isFloat() && "unexpected dtype");
    APFloat storage(static_cast<double>(literal));
    scalar = SIMDAttr::get({storage, *dtype}, scalarType);
  } else {
    assert(dtype->isIntLike());
    APInt storage(dtype->getWidthInBits() == -1 ? /*index or uindex*/ 64
                                                : dtype->getWidthInBits(),
                  literal, dtype->isSInt());
    scalar = SIMDAttr::get({storage, *dtype}, scalarType);
  }

  // splat to the target type.
  return SIMDSplatAttr::get(target.getContext(), scalar, target);
}

TypedAttr KGEN::splatFloatLiteralToSIMD(double literal, SIMDType target) {
  return splatLiteralToSIMDImpl<double>(literal, target);
}
TypedAttr KGEN::splatIntLiteralToSIMD(uint64_t literal, SIMDType target) {
  return splatLiteralToSIMDImpl<uint64_t>(literal, target);
}
