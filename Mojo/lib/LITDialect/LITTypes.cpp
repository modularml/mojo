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

#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LITDialect
//===----------------------------------------------------------------------===//

static ParseResult
parseStructTypeParams(AsmParser &p, SmallVectorImpl<TypedAttr> &paramValues);

void LITDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Mojo/LITDialect/LITTypes.cpp.inc"
      >();

  auto *dialect = getContext()->getOrLoadDialect<KGENDialect>();
  dialect->registerMnemonicType<MetaType>();
  dialect->registerMnemonicType<TraitType>();
  dialect->registerMnemonicType<OriginType>();
  dialect->registerMnemonicType<OriginSetType>();

  // Register the StructType parser.
  getContext()->getLoadedDialect<KGENDialect>()->setSymbolTypeParser(
      [&](AsmParser &p, SymbolRefAttr symbol) -> FailureOr<Type> {
        // Symbol reference is type positions are implicitly !lit.struct types.
        SmallVector<TypedAttr> values;
        if (failed(parseStructTypeParams(p, values)))
          return failure();

        // FIXME: Shouldn't just work with fully bound types.
        auto typeSig = TypeSignatureType::get(p.getContext());
        return StructType::get(symbol, values, typeSig);
      });
}

//===----------------------------------------------------------------------===//
// TypeSignatureType
//===----------------------------------------------------------------------===//

/// TODO: remove these?
static ParseResult parseTypeSignature(AsmParser &p,
                                      SmallVectorImpl<Type> &paramTypes,
                                      PogListAttr &paramListAttrs) {
  if (parseOptionalParamSignature(p, paramTypes, paramListAttrs, {}))
    return failure();
  return success();
}

static void printTypeSignature(AsmPrinter &p, ArrayRef<Type> paramTypes,
                               PogListAttr paramListAttrs) {
  printOptionalParamSignature(p, paramTypes, paramListAttrs);
}

LogicalResult
TypeSignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<Type> paramTypes,
                          PogListAttr paramListAttrs) {
  if (paramListAttrs.size() != paramTypes.size()) {
    return emitError() << "number of parameters doesn't match number of input "
                          "parameter types";
  }

  return success();
}

TypeSignatureType TypeSignatureType::remapToSignature(
    function_ref<InFlightDiagnostic()> emitError, ParamDeclArrayAttr paramDecls,
    PogListAttr paramListAttrs) {
  IndexRefRemapper remapper(paramDecls, {});
  SmallVector<Type> inputParamTypes =
      llvm::map_to_vector(paramDecls, [&](ParamDeclAttr decl) {
        return remapper.replace(decl.getType());
      });

  MLIRContext *ctx = paramDecls.getContext();
  SmallVector<ConstraintAttr> bodyConstraints;
  for (ConstraintAttr constraint : paramListAttrs.getBodyConstraints()) {
    bodyConstraints.push_back(
        cast<ConstraintAttr>(remapper.replace(constraint)));
  }

  paramListAttrs = PogListAttr::get(
      ctx, remapper.replace(paramListAttrs.getPogs()), bodyConstraints,
      paramListAttrs.getOrigVariadicConvention());
  return TypeSignatureType::getChecked(emitError, ctx, inputParamTypes,
                                       paramListAttrs);
}

TypeSignatureType TypeSignatureType::get(MLIRContext *context) {
  return get(context, /*paramTypes=*/{}, PogListAttr::get(context));
}

StringAttr TypeSignatureType::getParamName(size_t idx) const {
  return getParamListAttrs().getName(idx);
}

/// Bind parameter values to the signature, returning a new one.
TypeSignatureType TypeSignatureType::bind(ArrayRef<TypedAttr> values) const {
  assert(values.size() == getParamTypes().size() &&
         "expected full value set with UnboundAttrs for missing ones");

  PogListAttr paramListAttr = getParamListAttrs();

  SmallVector<Type> newParamTypes;
  SmallVector<PogMetadataAttr> newPogs;

  bool hasVarArg = false;
  ParameterEvaluator evaluator;
  for (auto [i, val, type, pogAttr] :
       llvm::enumerate(values, getParamTypes(), paramListAttr.getPogs())) {
    // If the current value is bound and we have a specified value, use it.
    if (!::isa<UnboundAttr>(val)) {
      evaluator.appendIndexBinding(val);
      continue;
    }

    // Otherwise it is still unbound, maintain it as such.
    newParamTypes.push_back(evaluator.getReboundType(type));
    newPogs.push_back(
        cast<PogMetadataAttr>(evaluator.getReboundAttribute(pogAttr)));
    hasVarArg |= pogAttr.isAnyVarArg();

    evaluator.appendIndexBinding(
        ParamIndexRefAttr::get(newParamTypes.size() - 1, newParamTypes.back()));
  }
  ArgConvention origVariadicConvention =
      hasVarArg ? paramListAttr.getOrigVariadicConvention()
                : ArgConvention::ByRefError;

  SmallVector<ConstraintAttr> newBodyConstraints;
  for (ConstraintAttr constraint : paramListAttr.getBodyConstraints()) {
    newBodyConstraints.push_back(
        cast<ConstraintAttr>(evaluator.getReboundAttribute(constraint)));
  }

  auto paramListAttrs = PogListAttr::get(
      getContext(), newPogs, newBodyConstraints, origVariadicConvention);
  return TypeSignatureType::get(getContext(), newParamTypes, paramListAttrs);
}

//===----------------------------------------------------------------------===//
// ModuleType
//===----------------------------------------------------------------------===//

OptionalParseResult ModuleType::parseValue(AsmParser &p,
                                           TypedAttr &value) const {
  return {}; // No custom parsing.
}

LogicalResult ModuleType::printValue(AsmPrinter &p, TypedAttr value) const {
  return failure(); // No custom printing.
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

MetaType LIT::StructType::getMetaType() { return MetaType::get(*this); }

OptionalParseResult LIT::StructType::parseValue(AsmParser &p,
                                                TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return {};

  // Handle `{}`.
  if (succeeded(p.parseOptionalRBrace())) {
    value = LITStructAttr::get({}, *this);
    return mlir::success();
  }

  // Special-case `{<value>}`.
  std::string name;
  if (failed(p.parseOptionalKeywordOrString(&name))) {
    TypedAttr element;
    if (parseColonTypeParamValue(p, element))
      return failure();
    value = LITStructAttr::get(
        {{StringAttr::get(p.getContext(), "_mlir_value"), element}}, *this);
    return p.parseRBrace();
  }

  // `parseOptionalKeywordOrString` consumed `name` as a potential field name,
  // but if the next token is `}` it is actually a bare ParamDeclRefAttr in the
  // `{<value>}` shorthand. This happens when the value has index type: the
  // printer omits the `:index` prefix (index is the default), leaving a bare
  // identifier indistinguishable from a field name without this lookahead.
  if (succeeded(p.parseOptionalRBrace())) {
    value = LITStructAttr::get(
        {{StringAttr::get(p.getContext(), "_mlir_value"),
          ParamDeclRefAttr::get(name, p.getBuilder().getIndexType())}},
        *this);
    return mlir::success();
  }

  // Parse `{(<name-type> = <value>)+}`.
  Type type;
  TypedAttr element;
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  auto parseElement = [&]() -> ParseResult {
    if (parseColonTypeOrIndex(p, type) || p.parseEqual() ||
        parseParamValue(p, element, type))
      return failure();
    values.emplace_back(StringAttr::get(p.getContext(), name), element);
    return success();
  };
  if (parseElement())
    return failure();
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseKeywordOrString(&name) || parseElement())
      return failure();
  }
  value = LITStructAttr::get(values, *this);

  return p.parseRBrace();
}

LogicalResult LIT::StructType::printValue(AsmPrinter &p,
                                          TypedAttr value) const {
  auto attr = ::dyn_cast<LITStructAttr>(value);
  if (!attr)
    return failure();
  ArrayRef<std::tuple<StringAttr, TypedAttr>> values = attr.getValues();

  p << '{';
  if (values.size() == 1 && std::get<0>(values.front()) == "_mlir_value" &&
      // Don't print 'add(x, y)', 'to/from_builtin(...)', or 'sugar_xxx(...)' as
      // the value, because the parser will think that is a field name.
      !::isa<ParamOperatorAttr, SugarAttr, CastFromBuiltinAttr,
             CastToBuiltinAttr>(std::get<1>(values.front()))) {
    printColonTypeParamValue(p, std::get<1>(values.front()));
  } else {
    llvm::interleaveComma(values, p, [&](const auto &element) {
      auto [name, value] = element;
      p.printKeywordOrString(name);
      printColonTypeOrIndex(p, value.getType());
      p << " = ";
      printParamValue(p, value);
    });
  }
  p << '}';
  return success();
}

LIT::StructType LIT::StructType::get(SymbolRefAttr name,
                                     ArrayRef<TypedAttr> paramValues,
                                     TypeSignatureType signature) {
  auto nameSym = SymbolAttr::get(name);

  // If this struct type has sugar, we compute a canonical version of it to cut
  // recursive walks during type canonicalization.
  bool anyDifferent = false;
  SmallVector<TypedAttr> canParams;
  canParams.reserve(paramValues.size());
  for (auto param : paramValues) {
    canParams.push_back(getCanonicalAttr(param));
    anyDifferent |= canParams.back() != param;
  }

  // This is canonical if all the parameters and signature are canonical. The
  // SymbolRefAttr is always canonical.
  StructType canonical;
  if (anyDifferent) {
    canonical = Base::get(name.getContext(), nameSym, canParams, signature,
                          /*canonical*/ StructType());
  }

  return Base::get(name.getContext(), nameSym, paramValues, signature,
                   canonical);
}

LIT::StructType LIT::StructType::get(MLIRContext *context, SymbolAttr name,
                                     ArrayRef<TypedAttr> paramValues,
                                     TypeSignatureType signature,
                                     StructType canonical) {
  // This method gets called by client doing general structural replacements,
  // e.g. a parameter with an arbitrary attribute.  This can turn canonical
  // forms to non-canonical and visa-versa, so always recompute the canonical
  // pointer.
  return get(name.getValue(), paramValues, signature);
}

LIT::StructType LIT::StructType::get(SymbolRefAttr name,
                                     TypeSignatureType signature) {
  return get(name, {}, signature);
}

SymbolRefAttr LIT::StructType::getSymbol() const {
  return getValue().getValue();
}

std::optional<StringRef> LIT::StructType::getAliasName(SymbolRefAttr symbol) {
  // Use the leaf name as the alias name.
  StringRef leaf = symbol.getLeafReference().getValue();
  unsigned offset = leaf.size();
  while (offset > 0 && std::isalnum(leaf[offset - 1]))
    --offset;
  if (offset == leaf.size() ||
      (!offset && symbol.getNestedReferences().empty()))
    return {};
  return leaf.substr(offset);
}

LogicalResult
LIT::StructType::verifySymbolUses(SymTabEvaluationContext &evaluationContext,
                                  Location loc) const {
  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

  DeclInterface decl = ::dyn_cast_or_null<DeclInterface>(
      symtab.lookupSymbolIn(module, getSymbol()));
  if (!decl) {
    return mlir::emitError(loc)
           << getSymbol() << " does not reference a KGEN type declaration";
  }

  if (getParamValues().empty() && decl.getInputParams().empty())
    return success();

  // We have to specialize the type's parameter decls.
  ParameterEvaluator evaluator(decl.getInputParams(), getParamValues());
  evaluator.setEvaluationContext(&evaluationContext);
  SmallVector<ParamDeclAttr, 8> specializedDecls;
  for (ParamDeclAttr decl : decl.getInputParams())
    specializedDecls.push_back(
        ::cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl)));

  return verifyParamDeclsMatch(
      "input parameter", "!lit.struct symbol use", getParamValues(), loc,
      getSymbol().getLeafReference(), specializedDecls, decl.getLoc());
}

static ParseResult
parseStructTypeParams(AsmParser &p, SmallVectorImpl<TypedAttr> &paramValues) {
  return parseParameterValues(p, paramValues);
}

static ParseResult
parseStructTypeBody(AsmParser &p, SymbolAttr &symbol,
                    SmallVectorImpl<TypedAttr> &paramValues) {
  if (p.parseCustomAttributeWithFallback(symbol) ||
      parseStructTypeParams(p, paramValues))
    return failure();
  return success();
}

mlir::OpAsmAliasResult LIT::StructType::getAlias(raw_ostream &os) const {
  // Simple case: Types with no parameters.
  std::optional<StringRef> symbolAliasName = getAliasName(getSymbol());
  ArrayRef<TypedAttr> paramValues = getParamValues();
  if (paramValues.empty()) {
    if (symbolAliasName) {
      os << *symbolAliasName;
      return mlir::OpAsmAliasResult::OverridableAlias;
    }
  }

  // Special case: SIMD types.
  if (symbolAliasName == "SIMD" && paramValues.size() == 2) {
    auto getSingleEltStructAttr = [&](TypedAttr value) -> TypedAttr {
      auto structAttr = dyn_cast<LITStructAttr>(value);
      if (structAttr && structAttr.getValues().size() == 1)
        return std::get<1>(structAttr.getValues().front());
      return {};
    };

    auto dtype = dyn_cast_if_present<KGEN::DTypeConstantAttr>(
        getSingleEltStructAttr(paramValues[0]));
    auto size = dyn_cast_if_present<IntegerAttr>(
        getSingleEltStructAttr(paramValues[1]));
    if (dtype && size) {
      if (dtype.getDType() == KGEN::KGENDType::index && size.getInt() == 1)
        os << "Int";
      else if (size.getInt() == 1)
        os << "Scalar_" << dtype.getDType();
      else
        os << "SIMD_" << dtype.getDType() << "_" << size.getInt();
      return mlir::OpAsmAliasResult::OverridableAlias;
    }
  }

  return mlir::OpAsmAliasResult::NoAlias;
}

static void printStructTypeBody(AsmPrinter &p, SymbolAttr symbol,
                                ArrayRef<TypedAttr> paramValues) {
  // Don't alias struct type with parameter references.  We will instead alias
  // the symbol attribute.
  //     We want "!Int" but "!lit.struct<#SIMD <...>>"
  if (!paramValues.empty() && succeeded(p.printAlias(symbol))) {
    // We must print a space here, because MLIR hard codes #FOO< syntax to be
    // an "extended dialect attribute" where FOO is a dialect.  See
    // mlir::Parser::parseExtendedAttr for more information.
    // TODO: Consider printing params with []'s.
    p << ' ';
  } else
    p << symbol.getValue();
  printParameterValues(p, paramValues);
}

/// Get the name of the referenced type, ignoring packages.
StringAttr LIT::StructType::getName() {
  auto symbol = getSymbol();
  if (symbol.getNestedReferences().empty())
    return symbol.getRootReference();
  return symbol.getNestedReferences().back().getAttr();
}

LIT::StructType LIT::StructType::bindAll(ArrayRef<TypedAttr> values) const {
  assert(getParamValues().size() == values.size() && "expected full value set");

  // The AnyStruct will have all of the parameters specified, e.g. something
  // like:
  // StructMetaType[Int : AnyType, UnboundAttr : I8, 42 : Int, UnboundAttr: F32]
  // but the TypeSignatureType will just have [I8, F32].  The input value
  // bindings must line up where they are already specified, but can further
  // refine the SignatureType.  See what to pass down to it.

  SmallVector<TypedAttr> newSignatureBindings;
  bool hadNewBinding = false;
  for (auto [cur, val] : llvm::zip(getParamValues(), values)) {
    // If the current value is bound, maintain it.
    if (!::isa<UnboundAttr>(cur)) {
      assert(cur == val && "cannot change bound parameter value");
    } else {
      hadNewBinding |= !::isa<UnboundAttr>(val);
      // Otherwise, propagate it into the TypeSignatureType.
      newSignatureBindings.push_back(val);
    }
  }

  // If we're refining our signature because we have new bindings, return an
  // AnyStruct with the updated signature and values.
  if (!hadNewBinding)
    return *this;

  auto newSig = getSignature().bind(newSignatureBindings);
  return LIT::StructType::get(getSymbol(), values, newSig);
}

LIT::StructType LIT::StructType::bindUnbound(ArrayRef<TypedAttr> values) const {
  SmallVector<TypedAttr> bindings;
  auto it = values.begin();
  for (TypedAttr value : getParamValues()) {
    if (::isa<UnboundAttr>(value))
      bindings.push_back(*it++);
    else
      bindings.push_back(value);
  }
  assert(it == values.end() && "expected all bindings to be consumed");
  return bindAll(bindings);
}

/// We don't sugar always_inline builtin calls to things that produce a literal,
/// we want things like "x+1" to fold to "5" when x is substituted with 4.
std::optional<SugarKind>
LIT::StructType::canElideSugarFor(TypedAttr attr) const {
  auto getTypeName = [&]() -> StringRef {
    if (auto structType = dyn_cast<LIT::StructType>(attr.getType()))
      return structType.getSymbol().getLeafReference().strref();
    return {};
  };

  // If the specified value is a struct with a single element, return the
  // element.
  auto getSingleEltStructAttr = [&](TypedAttr value) -> TypedAttr {
    auto structAttr = sugarDynCast<LITStructAttr>(value);
    if (!structAttr)
      return {};
    // If the struct has a single element, elide the braces.
    if (structAttr.getValues().size() == 1)
      return std::get<1>(structAttr.getValues().front());
    return {};
  };

  /// A StructAttr is due to an inline @always_inline("builtin") initializer.
  /// Elide it if we have the default type with a literal so we don't print
  /// Int(42), but print it if it is something weird like IntLiteral(42)
  if (auto elt = getSingleEltStructAttr(attr)) {
    auto typeName = getTypeName();
    if (typeName == "Int" || typeName == "UInt" || typeName == "SIMDLength")
      if (isa<IntegerAttr>(elt))
        return SugarKind::AlwaysInlineBuiltin;

    if (typeName == "SIMD")
      if (isa<KGEN::SIMDAttr>(elt))
        return SugarKind::AlwaysInlineBuiltin;

    if (typeName == "Bool" || typeName == "DType") {
      if (isa<SIMDAttr, IntegerAttr, DTypeConstantAttr>(elt))
        return SugarKind::Alias;
    }

    // Aggressively desugar address spaces with known values.
    if (typeName == "AddressSpace") {
      // We never need to sugar AddressSpace.GENERIC since AstPrinter knows
      // about this.  We special case this because it is so common and would
      // otherwise massively bloat the IR of all pointers and references.
      if (auto addrElt =
              sugarDynCastIfPresent<IntegerAttr>(getSingleEltStructAttr(elt))) {
        if (addrElt.getValue().isZero())
          return SugarKind::Alias;
      }
      return canElideSugarFor(elt);
    }
  }

  if (isa<LITStructAttr>(attr)) {
    StringRef typeName = getTypeName();
    if (typeName == "IntLiteral" || typeName == "FloatLiteral" ||
        typeName == "StringLiteral" || typeName == "Origin" ||
        typeName == "TypeList" || typeName == "ParameterList")
      return SugarKind::AlwaysInlineBuiltin;
  }

  return {};
}

Type LIT::StructType::getCachedCanonicalType(Type type) const {
  // Struct type has a canonical type cache to cut recursive walks.
  auto structType = cast<LIT::StructType>(type);
  if (auto can = structType.getCanonical())
    return can;
  return structType;
}

SymbolRefAttr LIT::StructType::getSymbolRef() const {
  return getValue().getValue();
}

//===----------------------------------------------------------------------===//
// StructMetaType
//===----------------------------------------------------------------------===//

static OptionalParseResult parseTypeValue(AsmParser &p, TypedAttr &value,
                                          Type metatype) {
  auto typeParser = [metatype](AsmParser &p,
                               Type &typeValue) -> OptionalParseResult {
    SymbolRefAttr symbol;
    OptionalParseResult result = p.parseOptionalAttribute(symbol);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      SmallVector<TypedAttr> values;
      if (parseParameterValues(p, values))
        return failure();
      TypeSignatureType typeSig;
      if (auto anyStruct = dyn_cast<StructMetaType>(metatype))
        typeSig = anyStruct.getSignature();
      else
        typeSig = TypeSignatureType::get(p.getContext());
      typeValue = LIT::StructType::get(symbol, values, typeSig);
    } else {
      result = parseOptionalKGENType(p, typeValue);
    }
    return result;
  };
  return parseSugaredTypeValue(p, value, metatype, typeParser);
}

static LogicalResult printTypeValue(AsmPrinter &p, TypedAttr value) {
  auto typePrinter = [](AsmPrinter &p, Type type) {
    if (auto ref = ::dyn_cast<LIT::StructType>(type)) {
      // Use the alias printer if suitable.
      if (failed(p.printAlias(ref))) {
        p << ref.getSymbol();
        printParameterValues(p, ref.getParamValues());
      }
    } else {
      printKGENType(p, type);
    }
  };
  return printSugaredTypeValue(p, value, typePrinter);
}

//===----------------------------------------------------------------------===//
// TraitType
//===----------------------------------------------------------------------===//
//
// TraitType supports conditional trait conformance by storing an optional
// parallel array of ConstraintAttrs alongside the trait symbols. Each
// constraint specifies the condition under which conformance to the
// corresponding trait applies.
//
// Design Decision: Trivially True Constraint Canonicalization
// ------------------------------------------------------------
// Rather than allowing null entries in the constraints array (which would
// complicate bytecode serialization), traits without explicit constraints use
// a "trivially true" constraint as a sentinel value. This constraint has a
// proposition of constant 1 (true).
//
// IMPORTANT: Trivially true constraints will NOT be printed in the textual IR.
// `@Trait where true` is semantically equivalent to `@Trait`, so we normalize
// to the simpler form. This means:
//
//   Input:  !lit.trait<@Foo where #kgen.constraint<1 : i1, loc("file":1:1)>>
//   Output: !lit.trait<@Foo>
//
// The location information from user-written `where true` constraints will
// not be preserved in the textual representation, but the semantic meaning
// is unchanged.
//
//===----------------------------------------------------------------------===//

TraitType TraitType::canonicalizeAndGet(MLIRContext *context,
                                        ArrayRef<TraitSymbolAttr> symbols,
                                        ArrayRef<ConstraintAttr> constraints) {
  // Fast path: no constraints provided means unconditional conformance.
  if (constraints.empty())
    return Base::get(context, symbols, constraints);

  // Canonicalize constraints:
  // 1. When all remaining constraints are trivially true, clear the array
  SmallVector<TraitSymbolAttr> canonSymbols;
  SmallVector<ConstraintAttr> canonConstraints;
  canonSymbols.reserve(symbols.size());
  canonConstraints.reserve(constraints.size());

  ConstraintAttr unconditional = getUnconditionalConstraint(context);
  bool hasAnyNonTrivialConstraint = false;

  for (auto [symbol, constraint] : llvm::zip(symbols, constraints)) {
    // Skip slots with false constraints - they are never satisfiable.
    if (isTriviallyFalseConstraint(constraint))
      continue;

    canonSymbols.push_back(symbol);

    if (isTriviallyTrueConstraint(constraint)) {
      canonConstraints.push_back(unconditional);
    } else {
      canonConstraints.push_back(constraint);
      hasAnyNonTrivialConstraint = true;
    }
  }

  // If all constraints are trivially true, clear the constraints array.
  if (!hasAnyNonTrivialConstraint)
    canonConstraints.clear();

  return Base::get(context, canonSymbols, canonConstraints);
}

Type TraitType::parse(AsmParser &p) {
  if (p.parseLess())
    return {};

  SmallVector<TraitSymbolAttr> symbols;
  SmallVector<ConstraintAttr> constraints;
  bool hasAnyConstraints = false;

  // Parse optional comma-separated list of trait symbols, each optionally
  // followed by "where <constraint>".
  // Format: @TraitA, @TraitB where <constraint>, @TraitC
  if (failed(p.parseOptionalGreater())) {
    auto parseTrait = [&]() -> ParseResult {
      TraitSymbolAttr symbol;
      if (parseTraitSymbol(p, symbol))
        return failure();
      symbols.push_back(symbol);

      // Check for optional "where <constraint>" after this symbol.
      if (succeeded(p.parseOptionalKeyword("where"))) {
        ConstraintAttr constraint;
        if (p.parseAttribute(constraint))
          return failure();
        constraints.push_back(constraint);
        hasAnyConstraints = true;
      } else {
        // No constraint for this trait - use placeholder for unconditional.
        constraints.push_back(getUnconditionalConstraint(p.getContext()));
      }
      return success();
    };

    if (parseTrait())
      return {};
    while (succeeded(p.parseOptionalComma())) {
      if (parseTrait())
        return {};
    }

    if (p.parseGreater())
      return {};
  }

  // If no constraints were specified at all, pass empty array.
  // The builder will handle canonicalization.
  if (!hasAnyConstraints)
    constraints.clear();

  // The builder handles all canonicalization (false constraint removal,
  // true constraint normalization, etc.)
  return TraitType::get(p.getContext(), symbols, constraints);
}

void TraitType::print(AsmPrinter &p) const {
  ArrayRef<TraitSymbolAttr> symbols = getSymbols();
  ArrayRef<ConstraintAttr> constraints = getConstraints();

  p << '<';
  for (size_t i = 0; i < symbols.size(); ++i) {
    if (i > 0)
      p << ", ";
    printTraitSymbol(p, symbols[i]);
    // Print "where <constraint>" for conditional conformance constraints.
    if (i < constraints.size() && !isTriviallyTrueConstraint(constraints[i])) {
      p << " where ";
      p.printAttribute(constraints[i]);
    }
  }
  p << '>';
}

LogicalResult TraitType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<TraitSymbolAttr> symbols,
                                ArrayRef<ConstraintAttr> constraints) {
  // If constraints are present, they must form a parallel array with symbols.
  if (!constraints.empty() && constraints.size() != symbols.size()) {
    return emitError() << "constraints array size (" << constraints.size()
                       << ") must match symbols array size (" << symbols.size()
                       << ")";
  }
  return success();
}

OptionalParseResult TraitType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult TraitType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value);
}

/// Return the metatype for this this trait as a value.
MetaType TraitType::getMetaType() { return MetaType::get(*this); }

/// Return symbols carried by this trait type.
ArrayRef<TraitSymbolAttr> TraitType::getTraitSymbols() const {
  return getSymbols();
}

/// Return a TypeParamAttr for a reference to this trait as a value, e.g.
/// uttering 'Stringable' in code.
TypedAttr TraitType::getPValue() {
  // Conditional trait conformance is a property of struct declarations, not of
  // trait values themselves. When a trait is referenced as a value (e.g.,
  // uttering 'Stringable' in code), it represents the unconditional trait type.
  // The constraints are only meaningful in the context of a struct's canonical
  // trait list where they specify conditions for conformance.
  assert(!hasConstraints() &&
         "cannot convert a conditional trait type to a pvalue - conditional "
         "conformance is a struct declaration property, not a trait value");
  return TypeParamAttr::get(*this, getMetaType());
}

// Sugar support: non-parameterized types get aliases.
mlir::OpAsmAliasResult LIT::TraitType::getAlias(raw_ostream &os) const {
  ArrayRef<TraitSymbolAttr> symbols = getSymbols();
  if (symbols.empty())
    return mlir::OpAsmAliasResult::NoAlias;
  SmallVector<StringRef> names;
  for (TraitSymbolAttr symbol : symbols) {
    if (std::optional<StringRef> name =
            StructType::getAliasName(symbol.getSymbol()))
      names.push_back(*name);
    else
      return mlir::OpAsmAliasResult::NoAlias;
  }
  // Conditional trait types get a "constrained_" prefix.
  if (hasConstraints())
    os << "constrained_";
  llvm::interleave(names, os, "_");
  return mlir::OpAsmAliasResult::OverridableAlias;
}

//===----------------------------------------------------------------------===//
// MetaType
//===----------------------------------------------------------------------===//

MetaType MetaType::getMetaType() {
  // We, currently, only have 2 level of meta type, i.e.,
  //
  //  type_of(struct)->meta<struct>,
  //  type_of(type_of(struct))->meta<meta<struct>>
  //
  // If the depth goes beyond 2, they will share the same indistinguishable meta
  // type `!kgen.type`, as the common meta type. I.e.,
  //
  //  type_of(type_of(type_of(....(struct)))) -> !kgen.type
  //
  // Making depth infinite is more correct, but practically having a max
  // depth of 2 is sufficient most of the time.
  if (sugarIsa<MetaType>(getType()))
    return MetaType();
  return MetaType::get(*this);
}

OptionalParseResult MetaType::parseValue(AsmParser &p, TypedAttr &value) const {
  return parseTypeValue(p, value, *this);
}

LogicalResult MetaType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printTypeValue(p, value);
}

mlir::OpAsmAliasResult MetaType::getAlias(raw_ostream &os) const {
  if (auto meta = dyn_cast<StructMetaType>(*this)) {
    // Don't alias metatypes that have parameter values, we want to alias things
    // like !mt_Int but not things like SIMD.  We'll alias the symbol instead.
    if (!meta.getParamValues().empty())
      return mlir::OpAsmAliasResult::NoAlias;
    if (std::optional<StringRef> name =
            StructType::getAliasName(meta.getSymbol())) {
      os << "mt_" << *name;
      return mlir::OpAsmAliasResult::OverridableAlias;
    }
  }
  return mlir::OpAsmAliasResult::NoAlias;
}

//===----------------------------------------------------------------------===//
// OriginType
//===----------------------------------------------------------------------===//

OptionalParseResult OriginType::parseValue(AsmParser &p,
                                           TypedAttr &result) const {
  // If there are any postfix origin syntax (<whatever>.field1.field2), then
  // parse them into 'result'.
  auto processPostFix = [&]() -> OptionalParseResult {
    if (!result)
      return failure();
    while (true) {
      if (succeeded(p.parseOptionalArrow())) {
        StringRef fieldName;
        if (failed(p.parseKeyword(&fieldName)))
          return failure();
        result = OriginFieldAttr::get(
            result, StringAttr::get(p.getContext(), fieldName));
        continue;
      }
      if (succeeded(p.parseOptionalLSquare())) {
        TypedAttr userName;
        if (failed(parseStringParam(p, userName)) || p.parseRSquare())
          return failure();
        result = InteriorOriginAttr::get(result, userName);
        continue;
      }
      // Otherwise, not a postfix thing.
      break;
    }
    return mlir::success();
  };

  // Parse |...| as OriginSet and OriginSetUnion.
  if (succeeded(p.parseOptionalVerticalBar())) {
    TypedAttr set;
    if (parseParamValue(p, set, OriginSetType::get(p.getContext())) ||
        p.parseVerticalBar())
      return failure();
    result = OriginSetUnionAttr::get(set, *this);
    return mlir::success();
  }

  // Handle names, and index references.
  if (succeeded(p.parseOptionalStar())) {
    std::string str;
    // Resolve ambiguity with *"...".
    if (succeeded(p.parseOptionalString(&str))) {
      result = ParamDeclRefAttr::get(str, *this);
      return processPostFix();
    }

    // Try to parse *(0,0) as an index reference.
    if (succeeded(p.parseOptionalLParen())) {
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      result = ParamIndexRefAttr::get(depth, index, *this);
      return processPostFix();
    }

    // *[x,y] is an ImplicitOriginRefAttr.
    size_t depth, index;
    if (succeeded(p.parseOptionalLSquare())) {
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRSquare())
        return failure();
      result = ImplicitOriginRefAttr::get(depth, index, *this);
      return processPostFix();
    }

    // We don't support *?
    p.emitError(p.getCurrentLocation(), "unknown origin value");
    return failure();
  }

  // Handle unions as comma separated elements in braces.
  if (succeeded(p.parseOptionalLBrace())) {
    SmallVector<TypedAttr> elements;
    // Body is {} or {elts}
    if (failed(p.parseOptionalRBrace())) {
      if (p.parseCommaSeparatedList(
              AsmParser::Delimiter::None,
              [&]() {
                elements.push_back({});
                return KGEN::parseParamValue(p, elements.back(), *this);
              },
              "in origin union") ||
          p.parseRBrace())
        return failure();
    }
    result = OriginUnionAttr::get(elements, *this);
    return processPostFix();
  }

  // Handle mutability casts in parens.
  if (succeeded(p.parseOptionalLParen())) {
    TypedAttr operand;
    if (p.parseKeyword("mutcast") || parseOriginParamValue(p, operand) ||
        p.parseRParen())
      return failure();
    result = OriginMutCastAttr::get(operand, *this);
    return processPostFix();
  }

  // Handle other things like param expressions, and KGEN operators. Disable
  // the type parser so we don't recurse.
  if (failed(parseParamValue(p, result, *this, /*disableTypeParser*/ true)))
    return {};
  return processPostFix();
}

LogicalResult OriginType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto declRef = ::dyn_cast<ParamDeclRefAttr>(value)) {
    printParamName(p, declRef.getName(), /*isRef*/ false);
    return success();
  }

  if (auto set = ::dyn_cast<OriginSetUnionAttr>(value)) {
    p << '|';
    printParamValue(p, set.getValue());
    p << '|';
    return success();
  }

  if (auto ref = ::dyn_cast<ImplicitOriginRefAttr>(value)) {
    p << "*[" << ref.getDepth() << ',' << ref.getIndex() << ']';
    return success();
  }

  if (auto unionAttr = ::dyn_cast<OriginUnionAttr>(value)) {
    p << '{';
    if (unionAttr.getNumOperands()) {
      printParamValue(p, unionAttr.getOperand(0));
      for (auto operand : unionAttr.getOperands().drop_front()) {
        p << ", ";
        printParamValue(p, operand);
      }
    }
    p << '}';
    return success();
  }

  if (auto mutcast = ::dyn_cast<OriginMutCastAttr>(value)) {
    p << "(mutcast ";
    printOriginParamValue(p, mutcast.getOperand());
    p << ")";
    return success();
  }

  // Print field access with dot notation.
  if (auto field = ::dyn_cast<OriginFieldAttr>(value)) {
    if (failed(printValue(p, field.getBase())))
      return failure();
    // FIXME: This should use ".field" instead of "->field" but MLIR doesn't
    // make it easy to parse a dot.
    p << "->";
    printParamName(p, field.getField(), /*isRef*/ false);
    return success();
  }

  // Print interior access with x[<string-param>] notation.
  if (auto interior = ::dyn_cast<InteriorOriginAttr>(value)) {
    if (failed(printValue(p, interior.getBase())))
      return failure();
    p << "[";
    printParamValue(p, interior.getUserName());
    p << "]";
    return success();
  }

  return failure();
}

OriginType OriginType::get(TypedAttr isMutable) {
  assert(KGEN::isScalarOf<KGENDType::kBool>(isMutable.getType()) &&
         "isMutable bit should scalar<bool>");
  return get(isMutable.getContext(), isMutable);
}

OriginType OriginType::get(MLIRContext *ctx, bool isMutable) {
  return get(ctx, SIMDAttr::getScalarBool(ctx, isMutable));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool OriginType::isMutableKnown(bool value) {
  if (auto cst = ::dyn_cast<SIMDAttr>(getCanonicalAttr(getIsMutable())))
    return cst.getAsBool() == value;
  return false;
}

/// Classify the mutability into Mutable/Immutable/Parametric.
OriginType::MutabilityClass OriginType::getMutabilityClass() {
  auto cst = ::dyn_cast<SIMDAttr>(getIsMutable());
  if (!cst)
    return Parametric;
  return cst.getAsBool() ? Mutable : Immutable;
}

/// Given a value of origin type, return true if the origin is known to have
/// the specified mutability.
bool OriginType::isMutableKnown(TypedAttr originValue, bool value) {
  return sugarCast<OriginType>(originValue.getType()).isMutableKnown(value);
}

/// Remove any OriginMutCast and Rebind if present.
TypedAttr OriginType::stripMutCastAndRebind(TypedAttr origin) {
  if (auto rebind = sugarDynCast<ParamOperatorAttr>(origin);
      rebind && rebind.getOpcode() == POC::Rebind)
    return stripMutCastAndRebind(rebind.getOperand(0));

  // Ignore MutCasts.
  if (auto mutCast = sugarDynCast<OriginMutCastAttr>(origin))
    return stripMutCastAndRebind(mutCast.getOperand());

  assert(isa<OriginType>(origin.getType()));
  return origin;
}

std::optional<SugarKind> OriginType::canElideSugarFor(TypedAttr attr) const {
  // Sugar for !lit.origin<true> and !lit.origin<false> can be elided, as these
  // print as MutableOrigin / ImmutableOrigin.  This ends up being a lot nicer
  // than: "origin_of(_lit_mut_cast[True, MutAnyOrigin].result".  We keep sugar
  // if our mutability so parametric expression.
  if (sugarIsa<SIMDAttr>(getIsMutable()))
    return SugarKind::Alias;

  // ImmStaticOrigin can also be elided.
  if (auto originField = sugarDynCast<OriginFieldAttr>(attr)) {
    if (isa<StaticOriginAttr>(originField.getBase())) {
      if (originField.getField().str() == "__constants__" &&
          originField.getType().isMutableKnown(false))
        return SugarKind::Alias;
    }
  }

  return {};
}

Type OriginType::getCachedCanonicalType(Type type) const { return {}; }

//===----------------------------------------------------------------------===//
// OriginSetType
//===----------------------------------------------------------------------===//

OptionalParseResult OriginSetType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  SmallVector<TypedAttr> origins;
  OptionalParseResult result = parseOptionalOriginSet(p, origins);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    value = OriginSetAttr::get(getContext(), origins, *this);
    return mlir::success();
  }
  return std::nullopt;
}

LogicalResult OriginSetType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto set = ::dyn_cast<OriginSetAttr>(value)) {
    printOriginSet(p, set.getOperands());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

RefType RefType::get(Type elementType, TypedAttr origin, TypedAttr addrSpace) {
  assert(sugarIsa<OriginType>(origin.getType()));
  return get(origin.getContext(), elementType, origin, addrSpace);
}

RefType RefType::get(Type elementType, TypedAttr origin, unsigned addrSpace) {
  auto *ctx = elementType.getContext();
  return get(elementType, origin,
             IntegerAttr::get(IndexType::get(ctx), addrSpace));
}

/// Return the pointer type that corresponds to this reference type, ignoring
/// the origin and the mutability.
PointerType RefType::getAsPointerType() {
  return PointerType::get(getElementType(), getAddressSpace());
}

/// Return this RefType but with a different element type.
RefType RefType::getWithElement(Type newElement) {
  return get(newElement, getOrigin(), getAddressSpace());
}

/// Return this RefType but with a different origin.
RefType RefType::getWithOrigin(TypedAttr newOrigin) {
  return get(getElementType(), newOrigin, getAddressSpace());
}

/// Return this RefType but with a different mutability.
RefType RefType::getWithMutability(bool isMut) {
  return get(getElementType(), OriginMutCastAttr::get(getOrigin(), isMut),
             getAddressSpace());
}

/// Return this RefType but with a different address space.
RefType RefType::getWithAddressSpace(TypedAttr newAddressSpace) {
  return get(getElementType(), getOrigin(), newAddressSpace);
}

/// Return the type of the origin reference, which is always a
/// `!lit.origin<mutability>` type.
OriginType RefType::getOriginType() {
  return sugarCast<OriginType>(getOrigin().getType());
}

/// Return a reference to the specified element type and mutability with
/// #lit.any.origin.
RefType RefType::getAnyOrigin(Type elementType, bool isMut,
                              TypedAttr addrSpace) {
  return get(elementType, AnyOriginAttr::get(elementType.getContext(), isMut),
             addrSpace);
}

RefType RefType::getAnyOrigin(Type elementType, bool isMut,
                              unsigned addrSpace) {
  return getAnyOrigin(
      elementType, isMut,
      IntegerAttr::get(IndexType::get(elementType.getContext()), addrSpace));
}

/// Return true if the mutable attribute is known to be the specific
/// constant.  This returns false if parametric or if the other value.
bool RefType::isMutableKnown(bool value) {
  return OriginType::isMutableKnown(getOrigin(), value);
}

/// Classify the mutability into Mutable/Immutable/Parametric.
OriginType::MutabilityClass RefType::getMutabilityClass() {
  return sugarCast<OriginType>(getOrigin().getType()).getMutabilityClass();
}

/// Return a (possibly parametric) specification for whether this reference
/// is a mutation or a read.
TypedAttr RefType::isMutable() {
  return sugarCast<OriginType>(getOrigin().getType()).isMutable();
}

/// Return true if this is in address space 0.
bool RefType::isDefaultAddrSpace() {
  auto addrSpace = getAddressSpace();
  if (!isa<IntegerAttr>(addrSpace))
    addrSpace = getCanonicalAttr(addrSpace);
  if (auto intAttr = ::dyn_cast<IntegerAttr>(addrSpace))
    return intAttr.getInt() == 0;

  return false;
}

/// Given an argument type+convention, if the convention has an implicit
/// reference, remove it from the type.
Type RefType::stripRefConvention(Type type, ArgConvention convention) {
  if (hasAddress(convention))
    return cast<RefType>(type).getElementType();
  return type;
}

OptionalParseResult RefType::parseValue(AsmParser &p, TypedAttr &value) const {
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

LogicalResult RefType::printValue(AsmPrinter &p, TypedAttr value) const {
  // Print a `store_to_mem` directive.
  if (auto memAttr = ::dyn_cast<StoreToMemAttr>(value)) {
    p << "store_to_mem(";
    printParamValue(p, memAttr.getValue());
    p << ')';
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// RefPackType
//===----------------------------------------------------------------------===//

RefPackType RefPackType::get(TypedAttr variadic, TypedAttr origin,
                             TypedAttr addressSpace) {
  return get(variadic.getContext(), variadic, origin, addressSpace);
}

ParamListAttr RefPackType::getVariadicIfResolved() const {
  return ::dyn_cast<ParamListAttr>(getVariadic());
}

/// Return the effective type (always a reference) of each element given
/// the type according to the type list.
RefType RefPackType::getElementRefTypeFor(Type elementType) {
  return RefType::get(elementType, getOrigin(), getAddressSpace());
}

/// This returns the element type of the variadic list parameter, typically
/// something like !kgen.type or a trait type.
Type RefPackType::getParamListElementType() {
  return ::cast<ParamListType>(getVariadic().getType()).getElementType();
}

//===----------------------------------------------------------------------===//
// REPLResultRefType
//===----------------------------------------------------------------------===//

REPLResultRefType REPLResultRefType::get(Type elementType) {
  auto *ctx = elementType.getContext();
  return get(ctx, elementType);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/LITDialect/LITTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// SignatureType Parsing
//===----------------------------------------------------------------------===//

static OptionalParseResult parseOptionalLITFuncType(AsmParser &p,
                                                    Type &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  size_t numOriginDecls = 0;
  if (succeeded(p.parseOptionalLSquare()))
    if (p.parseInteger(numOriginDecls) || p.parseRSquare())
      return failure();

  TypedAttr captureOrigins;
  auto originSet = OriginSetType::get(p.getContext());
  if (succeeded(p.parseOptionalColon())) {
    if (parseParamValue(p, captureOrigins, originSet) || p.parseColon())
      return failure();
  } else {
    captureOrigins = OriginSetAttr::get({}, originSet);
  }
  bool isNestedOriginsReadOnly =
      succeeded(p.parseOptionalKeyword("no_nested_origin_exclusivity"));
  bool definesInteriorOrigins =
      succeeded(p.parseOptionalKeyword("defines_interior_origins"));

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultValues;
  SmallVector<ArgConvention> argConventions;
  SmallVector<VariadicKind> argVariadics;
  std::optional<ArgConvention> origVariadicConvention;

  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    // Parse an optional argument name.
    if (parseOptionalName(p, argNames.emplace_back()))
      return failure();

    // Parse the argument type and its input convention.
    Type &type = argTypes.emplace_back();
    if (p.parseType(type) || parseConventionAndVariadicness(
                                 p, argConventions.emplace_back(),
                                 argVariadics.emplace_back(VariadicKind::None),
                                 origVariadicConvention, idx++))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type,
                                         hasAddress(argConventions.back()))))
      return failure();
    defaultValues.push_back(defaultVal);
    return success();
  };

  FunctionType functionType;
  FnEffects effects;
  OptionalParseResult result =
      parseOptionalSignatureValues(p, parseArg, functionType, effects,
                                   /*optionalResultList=*/false);
  if (!result.has_value())
    return std::nullopt;
  if (failed(*result))
    return failure();

  SmallVector<PassingKind> argPassingKinds;
  passingKindParser.populatePassingKinds(argPassingKinds);

  MLIRContext *ctx = p.getContext();
  auto pogList = PogListAttr::get(ctx, argNames, argPassingKinds, argVariadics,
                                  defaultValues, origVariadicConvention,
                                  /*bodyConstraints=*/{});
  auto metadata = FnMetaOriginDataAttr::get(ctx, numOriginDecls, captureOrigins,
                                            isNestedOriginsReadOnly,
                                            definesInteriorOrigins);
  signature =
      FuncType::getChecked([&] { return p.emitError(startLoc); }, functionType,
                           argConventions, effects, metadata, pogList);

  return success(!!signature);
}

static ParseResult parseLITFuncType(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalLITFuncType(p, signature);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(), "expected LIT signature");
}

//===----------------------------------------------------------------------===//
// GeneratorType Parsing
//===----------------------------------------------------------------------===//

/// Parses the LIT textual form (`!lit.generator<...>`) of a `GeneratorType`.
static ParseResult parseLITGenerator(AsmParser &p, Type &generator) {
  SmallVector<Type> inputParamTypes;
  PogListAttr paramListAttr = PogListAttr::get(p.getContext());
  Type body;
  auto parseBody = [&]() -> ParseResult {
    // Try to parse an unwrapped FnType fist.
    OptionalParseResult result = parseOptionalLITFuncType(p, body);
    if (result.has_value() && failed(*result))
      return failure();
    // If not a FnType, then parse as any other type.
    if (!result.has_value() && parseKGENType(p, body))
      return failure();
    return success();
  };

  if (KGEN::parseOptionalParamSignature(p, inputParamTypes, paramListAttr,
                                        parseBody))
    return failure();
  generator = GeneratorType::get(inputParamTypes, body, paramListAttr);
  return success();
}

Type LITDialect::parseType(DialectAsmParser &p) const {
  llvm::SMLoc typeLoc = p.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult = generatedTypeParser(p, &mnemonic, genType);

  if (parseResult.has_value())
    return genType;

  // Special alias for `!lit.fn` & `!lit.generator` types.
  if (mnemonic == "fn") {
    if (p.parseLess() || parseLITFuncType(p, genType) || p.parseGreater())
      return {};
    return genType;
  } else if (mnemonic == "generator") {
    if (p.parseLess() || parseLITGenerator(p, genType) || p.parseGreater())
      return {};
    return genType;
  }

  p.emitError(typeLoc) << "unknown type `" << mnemonic << "` in dialect `"
                       << getNamespace() << "`";
  return {};
}

void LITDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (succeeded(generatedTypePrinter(type, p)))
    return;
}

//===----------------------------------------------------------------------===//
// FuncType (LIT-dependent helpers)
//===----------------------------------------------------------------------===//

TypedAttr FuncType::getCaptureOrigins() {
  return ::cast<FnMetaOriginDataAttr>(getMetadata()).getCaptureOrigins();
}

bool FuncType::getIsNestedOriginsReadOnly() {
  return ::cast<FnMetaOriginDataAttr>(getMetadata())
      .getIsNestedOriginsReadOnly();
}

bool FuncType::getDefinesInteriorOrigins() {
  return ::cast<FnMetaOriginDataAttr>(getMetadata())
      .getDefinesInteriorOrigins();
}

size_t FuncType::getNumImplicitOriginDecls() {
  return ::cast<FnMetaOriginDataAttr>(getMetadata())
      .getNumImplicitOriginDecls();
}

Type FuncType::getUserResultType() {
  // If this function has a byref_result, return the reference element type.
  if (hasMemoryOnlyResult())
    return ::cast<RefType>(getArguments().back()).getElementType();
  return getResultType();
}

Type FuncType::getUserThrownType() {
  if (!isThrows())
    return {};
  auto numArgs = getArgConventions().size();
  assert(getArgConventions()[numArgs - 2] == ArgConvention::ByRefError &&
         "byref_error must be the second to last argument");
  return ::cast<RefType>(getArguments()[numArgs - 2]).getElementType();
}

FunctionType FuncType::substituteImplicitOriginsIntoValues(
    ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError) {
  assert(values.size() == getNumImplicitOriginDecls() &&
         "Incorrect # implicit origins specified");

  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // This checks to see if we found an ImplicitOriginRefAttr that's pointing
      // back to the current scope (the scope containing `values`), see
      // `IndexParameterReplacer` and
      if (auto ref = ::dyn_cast<ImplicitOriginRefAttr>(attr);
          ref && ref.getDepth() == depth) {
        if (ref.getIndex() >= values.size()) {
          emitError() << "implicit origin reference at depth " << depth
                      << " has an out-of-range index: " << ref.getIndex()
                      << " >= " << values.size();
          hadError = true;
          return ref;
        }
        return values[ref.getIndex()];
      }
      return nullptr;
    }

    ArrayRef<TypedAttr> values;
    function_ref<InFlightDiagnostic()> emitError;
    bool hadError = false;
  } substitutor;
  substitutor.values = values;
  substitutor.emitError = emitError;
  FunctionType result = substitutor.replace(getValues());
  return substitutor.hadError ? FunctionType() : result;
}

FuncType FuncType::getWithCaptureOrigins(TypedAttr origins) {
  return getWithMetadata(
      FnMetaOriginDataAttr::get(getContext(), getNumImplicitOriginDecls(),
                                origins, getIsNestedOriginsReadOnly(),
                                getDefinesInteriorOrigins()),
      getArgListAttrs());
}

Type FuncType::getIfVariadicListOrPack(size_t index) {
  if (!isPack(index) && !isPosVarArg(index))
    return {};

  // Look through references to the VariadicList/VariadicPack type.
  return RefType::stripRefConvention(getArgument(index),
                                     getArgConvention(index));
}

//===----------------------------------------------------------------------===//
// FnLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

FnLiteralTypeGeneratorType::FnLiteralTypeGeneratorType(GeneratorType gen)
    : FnTypeWrapperGeneratorType(gen) {
  assert((!gen || ::isa<FuncLiteralType>(gen.getBody())) &&
         "expected LIT generator wrapping FuncLiteralType");
}

FnLiteralTypeGeneratorType::FnLiteralTypeGeneratorType(
    FuncLiteralTypeGeneratorType gen)
    : FnTypeWrapperGeneratorType(gen) {
  assert((!gen || ::isa<FuncLiteralType>(gen.getBody())) &&
         "expected LIT generator wrapping FuncLiteralType");
}

PogListAttr FnLiteralTypeGeneratorType::getParamListAttrs() {
  return ::cast<PogListAttr>(GeneratorType::getParamListAttrs());
}

FuncLiteralType FnLiteralTypeGeneratorType::getBody() {
  return ::cast<FuncLiteralType>(GeneratorType::getBody());
}

bool FnLiteralTypeGeneratorType::classof(FuncLiteralTypeGeneratorType type) {
  return ::isa<FuncLiteralType>(type.getBody());
}

bool FnLiteralTypeGeneratorType::classof(Type type) {
  if (auto gen = ::dyn_cast<FuncLiteralTypeGeneratorType>(type))
    return classof(gen);
  return false;
}

//===----------------------------------------------------------------------===//
// FnTypeGeneratorType
//===----------------------------------------------------------------------===//

FnTypeGeneratorType::FnTypeGeneratorType(GeneratorType gen)
    : FnTypeWrapperGeneratorType(gen) {
  assert((!gen || ::isa<FuncType>(gen.getBody())) &&
         "expected generator of func type");
}

FnTypeGeneratorType::FnTypeGeneratorType(FuncTypeGeneratorType gen)
    : FnTypeWrapperGeneratorType(gen) {
  assert((!gen || ::isa<FuncType>(gen.getBody())) &&
         "expected generator of func type");
}

FuncType FnTypeGeneratorType::getBody() {
  return ::cast<FuncType>(GeneratorType::getBody());
}

PogListAttr FnTypeGeneratorType::getParamListAttrs() {
  return ::cast<PogListAttr>(GeneratorType::getParamListAttrs());
}

FnTypeGeneratorType
FnTypeGeneratorType::getWithCaptureOrigins(TypedAttr origins) {
  return getWithBody(getBody().getWithCaptureOrigins(origins));
}

/// This method replaces direct uses of NAMED implicit origin declarations
/// with index-based references.  originDecls specifies the names of the
/// implicit origin decls to replace.
///
/// depthOffset is subtracted from depth when set.
Type FnTypeGeneratorType::replaceImplicitOriginsWithIndexes(
    Type origType, ArrayRef<ParamDeclAttr> originDecls, size_t depthOffset) {

  // If there are no implicit origins, then this is a noop.
  if (originDecls.empty())
    return origType;

  // Replace named implicit origin parameter references with index-based
  // references in the signature.
  NameToImplicitOriginRefRemapper remapper(originDecls, depthOffset);
  return remapper.replace(origType);
}

/// This method replaces direct uses of NAMED implicit origin declarations
/// with index-based references corresponding to the signature. `originDecls`
/// specifies the names of the implicit origin decls.
FnTypeGeneratorType FnTypeGeneratorType::replaceImplicitOriginsWithIndexes(
    ArrayRef<ParamDeclAttr> originDecls) {
  assert(originDecls.size() == getBody().getNumImplicitOriginDecls() &&
         "Incorrect number of origin decls");
  return ::cast<FnTypeGeneratorType>(
      replaceImplicitOriginsWithIndexes(*this, originDecls, 1));
}

/// Reconstruct the signature using a list of named input parameters. These
/// parameters are prepended to the current signature and references are
/// remapped to index references. An additional array of indices corresponding
/// to variadic parameters of the prepended parameters is also required.
FnTypeGeneratorType FnTypeGeneratorType::prependParams(
    FnTypeGeneratorType sigGen, ArrayRef<ParamDeclAttr> parentParams,
    ArrayRef<StringAttr> paramNames, ArrayRef<TypedAttr> paramDefaults,
    ArrayRef<ConstraintAttr> bodyConstraints) {
  assert((paramNames.empty() || paramNames.size() == parentParams.size()) &&
         "paramNames, when provided, must match parent parameter list size");
  assert(
      (paramDefaults.empty() || paramDefaults.size() == parentParams.size()) &&
      "paramDefaults, when non-empty, must match parentParams");
  IndexRefRemapper remapper(parentParams, parentParams.size());
  IndexRefRemapper contextRemapper(parentParams, /*offset=*/0);
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sigGen.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  FuncType sig = sigGen.getBody();
  FnMetadataAttrInterface fnMetadata = remapper.replace(sig.getMetadata());

  SmallVector<StringAttr> names;
  if (paramNames.empty()) {
    names = llvm::map_to_vector(
        parentParams, [](ParamDeclAttr param) { return param.getName(); });
  } else {
    names = llvm::to_vector(paramNames);
  }

  PogListAttr oldMeta = remapper.replace(sigGen.getParamListAttrs());
  SmallVector<TypedAttr> remappedParamDefaults;
  for (TypedAttr defaultValue : paramDefaults) {
    remappedParamDefaults.push_back(
        defaultValue ? cast<TypedAttr>(contextRemapper.replace(defaultValue))
                     : TypedAttr());
  }
  SmallVector<ConstraintAttr> remappedBodyConstraints;
  for (ConstraintAttr constraint : bodyConstraints) {
    remappedBodyConstraints.push_back(
        cast<ConstraintAttr>(contextRemapper.replace(constraint)));
  }

  PogListAttr genMetadata = oldMeta.prependAsInferredParams(
      names, remappedParamDefaults, remappedBodyConstraints);

  PogListAttr remappedArgListAttrs;
  if (PogListAttr argListAttrs = sig.getArgListAttrs())
    remappedArgListAttrs = cast<PogListAttr>(remapper.replace(argListAttrs));
  return FuncTypeGeneratorType::get(
      inputParamTypes, remapper.replace(sig.getValues()),
      sig.getArgConventions(), sig.getFnEffects(), fnMetadata, genMetadata,
      remappedArgListAttrs);
}

bool FnTypeGeneratorType::classof(FuncTypeGeneratorType type) {
  return ::isa<FuncType>(type.getBody());
}

bool FnTypeGeneratorType::classof(Type type) {
  if (auto sig = ::dyn_cast<FuncTypeGeneratorType>(type))
    return classof(sig);
  return false;
}

//===----------------------------------------------------------------------===//
// MetaTypeOf
//===----------------------------------------------------------------------===//

SymbolRefAttr StructMetaType::getSymbol() const {
  return getType().getValue().getValue();
}

TypeSignatureType StructMetaType::getSignature() const {
  return getType().getSignature();
}

ArrayRef<TypedAttr> StructMetaType::getParamValues() const {
  return getType().getParamValues();
}

StructMetaType StructMetaType::bindAll(ArrayRef<TypedAttr> values) const {
  return StructMetaType::get(getType().bindAll(values));
}

StructMetaType StructMetaType::bindUnbound(ArrayRef<TypedAttr> values) const {
  return StructMetaType::get(getType().bindUnbound(values));
}

SymbolRefAttr StructMetaMetaType::getSymbol() const {
  return getType().getSymbol();
}

TypeSignatureType StructMetaMetaType::getSignature() const {
  return getType().getSignature();
}

ArrayRef<TypedAttr> StructMetaMetaType::getParamValues() const {
  return getType().getParamValues();
}

StructMetaMetaType
StructMetaMetaType::bindAll(ArrayRef<TypedAttr> values) const {
  return StructMetaMetaType::get(getType().bindAll(values));
}

StructMetaMetaType
StructMetaMetaType::bindUnbound(ArrayRef<TypedAttr> values) const {
  return StructMetaMetaType::get(getType().bindUnbound(values));
}

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

Type LIT::getSignatureUserResultType(FnOrFnLiteralTypeGeneratorType sigType,
                                     ArrayRef<Type> argTypes, Type resultType) {
  // If this function has a byref_result, return the reference element type.
  if (sigType.hasMemoryOnlyResult())
    return cast<RefType>(argTypes.back()).getElementType();
  return resultType;
}

/// If this specified operation is a call-like operation, return the
/// FnTypeGeneratorType for the callee, otherwise return null.
LIT::FnTypeGeneratorType LIT::getFnTypeFromCall(Operation &op) {
  if (auto directCall = dyn_cast<KGENCallOpInterface>(op))
    return directCall.getCalleeType();
  if (auto indirectCall = dyn_cast<LIT::CallIndirectOp>(op))
    return indirectCall.getCalleeType();
  return {};
}
