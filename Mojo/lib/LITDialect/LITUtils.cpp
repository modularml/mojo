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
// verifying LIT related operations and types.
//
//===----------------------------------------------------------------------===//

#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

bool LIT::isMetaType(Type type) {
  type = SugarAttr::strip(type);
  if (auto genType = dyn_cast<GeneratorType>(type))
    return isMetaType(genType.getBody());
  if (auto param = dyn_cast<ParamType>(type))
    return sugarIsa<StructMetaType, AnyTraitType>(param.getParam().getType());
  if (isa<NonStructTypeType, StructMetaType, TraitType, TypeType,
          FnLiteralTypeGeneratorMetaType>(type))
    return true;
  // A plain MetaType wrapping something already meta-classified is itself a
  // metatype too.
  if (auto meta = dyn_cast<MetaType>(type))
    return isMetaType(meta.getType());
  return false;
}
bool LIT::isVariadicOfMetaType(Type type) {
  auto va = sugarDynCast<ParamListType>(type);
  return va && LIT::isMetaType(va.getElementType());
}

bool LIT::isFirstLevelTypeExpr(TypedAttr typeExpr) {
  if (!typeExpr)
    return false;

  auto type = SugarAttr::strip(typeExpr.getType());
  if (auto param = dyn_cast<ParamType>(type)) {
    return sugarIsa<StructMetaMetaType, AnyTraitType,
                    FnLiteralTypeGeneratorMetaMetaType>(
        param.getParam().getType());
  }
  if (isa<StructMetaType, TraitType, NonStructTypeType,
          FnLiteralTypeGeneratorMetaType>(type))
    return true;

  // TypeType is not always a L1 type expression.
  return false;
}

TypedAttr LIT::getEmptyStructValue(Type type) {
  auto structType = sugarDynCast<StructType>(type);
  if (!structType)
    return {};

  // LITStructAttr's type must be a StructType, so a sugared type comes back
  // wrapped in a rebind.
  return ParamOperatorAttr::getRebind(LITStructAttr::get({}, structType), type);
}

bool LIT::isTypeExpr(TypedAttr attr) { return isMetaType(attr.getType()); }
bool LIT::isVariadicOfTypeExpr(TypedAttr attr) {
  auto va = sugarDynCast<ParamListAttr>(attr);
  return va && llvm::all_of(va.getValues(), LIT::isTypeExpr);
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Print a (potentially) parametric mutability specifier and then a value.  The
/// forms are: "imm expr", "mut expr", "mut=<expr>, expr" and "muttoimm expr"
/// without quotes.
void LIT::printOriginParamValue(AsmPrinter &p, TypedAttr value) {
  // If the type is sugared, then we don't want to sugar this operation because
  // round tripping would lose the sugar.
  if (auto castVal = dyn_cast<OriginMutCastAttr>(value)) {
    if (auto type = dyn_cast<OriginType>(value.getType())) {
      if (auto srcType = dyn_cast<OriginType>(castVal.getOperand().getType())) {
        // It is extremely common to have a OriginMutCastAttr cast from known
        // mutable origin to known immutable origin (this happens when borrowed
        // arguments are formed).  So much so that we sugar it.
        if (type.isMutableKnown(false) && srcType.isMutableKnown(true)) {
          p << "muttoimm ";
          // Now that the type is specified, print the origin value itself.
          printParamValue(p, castVal.getOperand());
          return;
        }
      }
    }
  }

  TypedAttr mutability = sugarCast<OriginType>(value.getType()).isMutable();
  if (auto boolAttr = dyn_cast<SIMDAttr>(mutability)) {
    p << (boolAttr.getAsBool() ? "mut " : "imm ");
  } else {
    p << "mut=";
    printParamValue(p, mutability);
    p << ", ";
  }

  // Now that the type is specified, print the origin value itself.
  printParamValue(p, value);
}

void OriginPrinter::printParam(raw_ostream &os, TypedAttr param) const {
  StreamAsmPrinter p(os);
  printParamValue(p, param, {});
}

void OriginPrinter::printDeclRef(raw_ostream &os,
                                 ParamDeclRefAttr declRef) const {
  if (isPrettyPrint())
    os << demangleParameterName(declRef.getName());
  else
    printAsMojoStringLiteral(declRef.getName(), os);
}

ParamDeclRefAttr
OriginPrinter::resolveIndexRef(raw_ostream &os,
                               ParamIndexRefAttr idxRef) const {
  return {};
}

std::optional<llvm::StringRef>
OriginPrinter::resolveImplicitOriginRef(raw_ostream &os,
                                        ImplicitOriginRefAttr originRef) const {
  return std::nullopt;
}

TypedAttr OriginPrinter::prepareSugarParam(raw_ostream &os,
                                           SugarAttr sugar) const {
  return sugar.getExpanded();
}

void OriginPrinter::print(raw_ostream &os, TypedAttr param,
                          bool elideOriginOf) const {
  if (auto originField = dyn_cast<OriginFieldAttr>(param)) {
    if (isa<StaticOriginAttr>(originField.getBase())) {
      if (originField.getField().str() == "__constants__" &&
          originField.getType().isMutableKnown(false)) {
        os << "ImmStaticOrigin";
        return;
      }
    }
  }

  if (auto originUnion = dyn_cast<OriginUnionAttr>(param)) {
    if (originUnion.getNumOperands() == 0) {
      if (originUnion.getType().isMutableKnown(true))
        os << "MutUntrackedOrigin";
      else if (originUnion.getType().isMutableKnown(false))
        os << "ImmUntrackedOrigin";
      else {
        os << "UntrackedOrigin[";
        printParam(os, originUnion.getType().getIsMutable());
        os << "]";
      }
      return;
    }

    if (!isPrettyPrint())
      os << '{';
    else if (!elideOriginOf)
      os << "origin_of(";

    llvm::interleaveComma(originUnion.getOperands(), os, [&](TypedAttr elt) {
      print(os, elt, /*elideOriginOf=*/true);
    });

    if (!isPrettyPrint())
      os << '}';
    else if (!elideOriginOf)
      os << ')';
    return;
  }

  if (auto mutcast = dyn_cast<OriginMutCastAttr>(param)) {
    if (isPrettyPrint())
      return print(os, mutcast.getOperand(), elideOriginOf);

    if (mutcast.getType().isMutableKnown(false))
      os << "(muttoimm ";
    else
      os << "(mutcast ";
    print(os, mutcast.getOperand(), elideOriginOf);
    os << ")";
    return;
  }

  if (auto anyOrig = dyn_cast<AnyOriginAttr>(param)) {
    if (anyOrig.getType().isMutableKnown(true))
      os << "MutUnsafeAnyOrigin";
    else if (anyOrig.getType().isMutableKnown(false))
      os << "ImmUnsafeAnyOrigin";
    else
      os << "SomeUnsafeAnyOrigin";
    return;
  }

  if (auto comptimeOrig = dyn_cast<ComptimeOriginAttr>(param)) {
    os << "ComptimeOrigin";
    return;
  }

  if (isa<UnknownAttr, UnboundAttr, SingletonAttr>(param)) {
    os << "_";
    return;
  }

  if (auto sugar = dyn_cast<SugarAttr>(param))
    return print(os, prepareSugarParam(os, sugar), elideOriginOf);

  if (auto poa = dyn_cast<ParamOperatorAttr>(param)) {
    assert(poa.getOpcode() == POC::Rebind && "unexpected operator");
    return print(os, poa.getOperand(0), elideOriginOf);
  }

  if (auto originRef = dyn_cast<ImplicitOriginRefAttr>(param)) {
    if (auto argName = resolveImplicitOriginRef(os, originRef)) {
      os << *argName;
      return;
    }
    os << "*[" << originRef.getDepth() << ',' << originRef.getIndex() << "]";
    return;
  }

  // Otherwise, this is a reference to a declaration or parameter.
  if (isPrettyPrint() && !elideOriginOf)
    os << "origin_of(";

  // RAII type to print the closing paren when this scope returns.
  struct ParenPrinter {
    raw_ostream &os;
    bool printParen;
    ~ParenPrinter() {
      if (printParen)
        os << ")";
    }
  };
  ParenPrinter parenGuard{os, isPrettyPrint() && !elideOriginOf};

  if (auto originField = dyn_cast<OriginFieldAttr>(param)) {
    print(os, originField.getBase(), /*elideOriginOf=*/true);
    os << '.' << originField.getField().str();
    return;
  }

  if (auto interior = dyn_cast<InteriorOriginAttr>(param)) {
    // Special case the interior origins that CheckLifetimes uses to model
    // subtree origins.
    if (auto name = dyn_cast<StringAttr>(interior.getUserName())) {
      if (name.strref() == subtreeInteriorOriginAttrName) {
        print(os, interior.getBase(), /*elideOriginOf=*/false);
        os << ".subtree";
        return;
      }
    }

    print(os, interior.getBase(), /*elideOriginOf=*/true);
    os << "[";
    printParam(os, interior.getUserName());
    os << "]";
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(param)) {
    printDeclRef(os, declRef);
    return;
  }

  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param)) {
    if (auto declRef = resolveIndexRef(os, indexRef))
      print(os, declRef, elideOriginOf);
    else
      printParam(os, param);
    return;
  }

  if (auto subtree = dyn_cast<OriginSubtreeAttr>(param)) {
    print(os, subtree.getBase(), /*elideOriginOf=*/false);
    os << ".subtree";
    return;
  }

  param.dump();
  llvm_unreachable("unknown origin parameter");
}

/// Given a subtree origin (a~), return an interior origin (a["magicname"]) that
/// we can use for analysis of an interior origin that might be contained within
/// it.  This is used by CheckLifetimes to model subtree origins.
InteriorOriginAttr LIT::getInteriorForSubtreeOrigin(OriginSubtreeAttr subtree) {
  auto result =
      InteriorOriginAttr::get(subtree.getBase(), subtreeInteriorOriginAttrName);
  // In general x["foo"] might not return an InteriorOrigin, because when x is a
  // union or mutcast, things get folded and rearranged. However, we know that
  // the input here is to something that already had been folded when forming
  // the subtree.
  return cast<InteriorOriginAttr>(result);
}

ParseResult LIT::parseOriginParamValue(AsmParser &p, TypedAttr &result) {
  OriginType type;
  // Parse the pretty type specifier if present.
  if (succeeded(p.parseOptionalKeyword("imm"))) {
    type = OriginType::get(p.getContext(), false);
  } else if (succeeded(p.parseOptionalKeyword("mut"))) {
    // !lit.ref<T, mut origin>    ==> mutable
    TypedAttr mutability;
    if (failed(p.parseOptionalEqual())) {
      mutability = SIMDAttr::getScalarBool(p.getContext(), true);
    } else {
      // !lit.ref<T, mut=expr, origin  ==> parametric
      if (parseScalarBoolParamValue(p, mutability) || p.parseComma())
        return failure();
    }
    type = OriginType::get(mutability);
  } else if (succeeded(p.parseOptionalKeyword("muttoimm"))) {
    // Operand is mutable, casted to immutable.
    if (KGEN::parseParamValue(p, result, OriginType::get(p.getContext(), true)))
      return failure();
    result = OriginMutCastAttr::get(result, false);
    return success();
  } else {
    // If none of "mut/imm/muttoimm" are specified, it may be an "ugly" style.
    // This is useful to support for Mojo composability.
    return p.parseAttribute(result);
  }

  // Ok, we found the type of the origin, parse the value next.
  return KGEN::parseParamValue(p, result, type);
}

void LIT::printNestedSymbolReference(raw_ostream &os, SymbolRefAttr symbol) {
  os << symbol.getRootReference().strref();
  for (FlatSymbolRefAttr nestedRef : symbol.getNestedReferences())
    os << "::" << nestedRef.getValue();
}

ParseResult LIT::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                           Type type, bool hasAddress) {
  if (hasAddress)
    if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementType();
  return KGEN::parseOptionalDefaultValue(p, defaultVal, type);
}

void LIT::printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal,
                                    Type type, bool hasAddress) {
  if (hasAddress)
    if (auto ref = dyn_cast<RefType>(type))
      type = ref.getElementType();
  KGEN::printOptionalDefaultValue(p, defaultVal, type);
}

ParseResult LIT::parseOriginSet(AsmParser &p,
                                SmallVectorImpl<TypedAttr> &lifetimes) {
  OptionalParseResult result = parseOptionalOriginSet(p, lifetimes);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(), "expected a '{'");
  return *result;
}

OptionalParseResult
LIT::parseOptionalOriginSet(AsmParser &p,
                            SmallVectorImpl<TypedAttr> &lifetimes) {
  if (failed(p.parseOptionalLBrace()))
    return std::nullopt;
  if (succeeded(p.parseOptionalRBrace()))
    return mlir::success();

  auto parseLifetime = [&]() -> ParseResult {
    TypedAttr mut;
    if (succeeded(p.parseOptionalKeyword("mut")))
      mut = SIMDAttr::getScalarBool(p.getContext(), true);
    else if (succeeded(p.parseOptionalKeyword("imm")))
      mut = SIMDAttr::getScalarBool(p.getContext(), false);
    else if (p.parseLParen() || parseScalarBoolParamValue(p, mut) ||
             p.parseRParen())
      return failure();
    return parseParamValue(p, lifetimes.emplace_back(), OriginType::get(mut));
  };
  if (p.parseCommaSeparatedList(parseLifetime))
    return failure();
  return p.parseRBrace();
}

void LIT::printOriginSet(AsmPrinter &p, ArrayRef<TypedAttr> lifetimes) {
  p << '{';
  auto printLifetime = [&](TypedAttr origin) {
    auto type = cast<OriginType>(origin.getType());
    TypedAttr mut = type.isMutable();
    // If the mutability is known, pretty print it. Otherwise, print the
    // parametric mutability expression within parens.
    if (auto known = dyn_cast<SIMDAttr>(mut)) {
      p << (known.getAsBool() ? "mut" : "imm");
    } else {
      p << '(';
      printParamValue(p, mut);
      p << ')';
    }
    p << ' ';
    printParamValue(p, origin);
  };
  llvm::interleaveComma(lifetimes, p, printLifetime);
  p << '}';
}

bool LIT::isEmptyOriginSet(TypedAttr attr) {
  if (!attr)
    return true;
  if (auto set = dyn_cast<OriginSetAttr>(attr))
    return set.getOperands().empty();
  return false;
}

void LIT::printFnType(AsmPrinter &p, FuncType signature) {
  FnMetaOriginDataAttr metadata =
      ::cast<FnMetaOriginDataAttr>(signature.getMetadata());
  if (unsigned numOriginDecls = metadata.getNumImplicitOriginDecls())
    p << '[' << numOriginDecls << ']';
  if (!isEmptyOriginSet(metadata.getCaptureOrigins())) {
    p << ':';
    printParamValue(p, metadata.getCaptureOrigins());
    p << ':';
  }
  if (signature.getIsNestedOriginsReadOnly())
    p << "no_nested_origin_exclusivity";
  if (signature.getDefinesInteriorOrigins()) {
    if (signature.getIsNestedOriginsReadOnly())
      p << ' ';
    p << "defines_interior_origins";
  }

  PogListAttr argListAttr = signature.getArgListAttrs();
  PassingKindPrinter passingKindPrinter(p, argListAttr, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(i);

    StringAttr argName = signature.getArgName(i);
    if (!argName.empty()) {
      p.printString(argName);
      p << ": ";
    }

    p << signature.getArgument(i);
    ArgConvention argConv = signature.getArgConvention(i);
    VariadicKind variadicness = argListAttr.getVariadicKind(i);
    if (variadicness == VariadicKind::PosVarArg ||
        variadicness == VariadicKind::PackVarArg) {
      assert(argConv == ArgConvention::ImmMem ||
             argConv == ArgConvention::Mut ||
             argConv == ArgConvention::OwnedMem ||
             argConv == ArgConvention::OwnedReg);
      argConv = signature.getVariadicConvention(i);
    }
    printConventionAndVariadicness(p, argConv, variadicness);
    printOptionalDefaultValue(p, argListAttr.getDefault(i),
                              signature.getArgument(i), hasAddress(argConv));

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(i);
  };

  printSignatureValues(p, printElt, signature.getValues(),
                       signature.getArgConventions(), signature.getFnEffects(),
                       /*optionalResultList=*/false);
  assert(argListAttr.getBodyConstraints().empty());
}

//===----------------------------------------------------------------------===//
// MangledSymbol
//===----------------------------------------------------------------------===//

MangledSymbol MangledSymbol::mangle(mlir::SymbolOpInterface op) {
  MangledSymbol out;
  // The parser mangles the argument types into the symbol name.
  size_t firstParen = op.getName().find('(');
  if (firstParen == std::string::npos)
    firstParen = op.getName().size();
  // Get the name of the func.
  out.symName =
      StringAttr::get(op.getContext(), op.getName().take_front(firstParen));
  out.identifier = StringAttr::get(
      op->getContext(),
      op.getName().take_front(op.getName().find_first_of("[(")));

  auto signatureStr =
      StringAttr::get(op.getContext(), op.getName().drop_front(firstParen));
  // If the operation is function-like, we can get its signature. However, using
  // it for name mangling breaks a lot of things right now.
  // TODO(10920): We have to re-evaluate if we want to have the parser doing
  //   some of this, or if we want to mangle it here.
  if (auto funcLike = dyn_cast<FuncInterface>(op.getOperation()))
    out.signature = funcLike.getFunctionType();
  else
    out.signature = nullptr;

  // Grab parent structs/modules/etc., add them in order from in -> out (they'll
  // be added to the name from out->in).
  Operation *parentOp = op;
  while ((parentOp = parentOp->getParentOp())) {
    TypeSwitch<Operation *>(parentOp)
        .Case([&](StructDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case([&](TraitDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case([&](ExtensionDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case<FileModuleOp, PackageOp>(
            [&](auto op) { out.moduleNames.push_back(op.getNameAttr()); });
  }
  std::reverse(out.structNames.begin(), out.structNames.end());
  std::reverse(out.moduleNames.begin(), out.moduleNames.end());

  std::string mangledName;
  llvm::raw_string_ostream nameStream(mangledName);
  // Emit the parent module and struct names. Module names are prefixed with `$`
  // - which provides a signal for what's a module vs struct when demangling.
  for (auto name : llvm::concat<StringAttr>(out.moduleNames, out.structNames))
    nameStream << name.getValue() << "::";
  // Finally, function name and argument types. Use the string coming out of the
  // parser rather than the actual function type.
  nameStream << out.symName.getValue() << signatureStr.getValue();

  out.mangled = StringAttr::get(op.getContext(), mangledName);
  return out;
}

/// Parse a mangled signature from `typeStr`. Expects a signature that looks
/// like `(type1,type2)rtype1,rtype2`.
static FailureOr<FunctionType> parseMangledSignature(MLIRContext *ctx,
                                                     StringRef typeStr) {
  SmallVector<Type> inputTypes, resultTypes;
  SmallVector<Type> *typeVec = &inputTypes;
  if (typeStr.empty())
    return FunctionType{};

  // Drop the first '(' if there is one.
  if (typeStr.starts_with("("))
    typeStr = typeStr.drop_front();

  // If the first thing in the string is the closing paren, move straight to
  // result types.
  if (typeStr.starts_with(")")) {
    typeStr = typeStr.drop_front();
    typeVec = &resultTypes;
  }

  // Now, parse the type string.
  while (!typeStr.empty()) {
    size_t numBytes = 0;
    Type t = mlir::parseType(typeStr, ctx, &numBytes);
    if (!t)
      return failure();

    typeVec->push_back(t);
    typeStr = typeStr.drop_front(numBytes);
    // Drop the comma.
    if (typeStr.starts_with(","))
      typeStr = typeStr.drop_front();

    // If we have reached the closing paren, then skip it and parse any
    // leftovers into the result types.
    if (typeStr.starts_with(")")) {
      typeStr = typeStr.drop_front();
      typeVec = &resultTypes;
    }
  }

  return FunctionType::get(ctx, inputTypes, resultTypes);
}

FailureOr<MangledSymbol> MangledSymbol::demangle(StringAttr mangled,
                                                 bool parseSignature) {
  MangledSymbol out;
  out.mangled = mangled;
  StringRef m = mangled.getValue();
  // We'll first tokenize the owning module and structs.
  size_t separator = m.find("::");
  size_t firstOpen = m.find_first_of("([");
  for (; separator != std::string::npos && separator < firstOpen;
       separator = m.find("::"), firstOpen = m.find_first_of("([")) {
    StringRef current = m.take_front(separator);
    // Drop until the separator.
    m = m.drop_front(separator);
    // Skip past the separator as well (if it exists).
    m.consume_front("::");
    // FIXME: Can't distinguish between struct and modules, but does it matter?
    out.moduleNames.push_back(StringAttr::get(mangled.getContext(), current));
  }
  // Get the name of the func and the types of its arguments.
  StringRef nameWithParameters = m.take_front(m.find('('));
  StringRef nameWithoutParameters = m.take_front(firstOpen);

  out.symName = StringAttr::get(mangled.getContext(), nameWithParameters);
  out.identifier = StringAttr::get(mangled.getContext(), nameWithoutParameters);

  size_t firstParen = m.find('(');
  if (firstParen == std::string::npos)
    firstParen = m.size();

  // If there's no parenthesis here, don't even parse out the signature.
  if (firstParen == m.size()) {
    out.signature = nullptr;
    return out;
  }

  // If there are more mangled symbols, then there are Mojo types we cannot
  // parse in general.
  if (separator != std::string::npos)
    return out;

  if (!parseSignature)
    return out;

  // If we *have* a signature, parse it out.
  FailureOr<FunctionType> sigOr =
      parseMangledSignature(mangled.getContext(), m.drop_front(firstParen));
  if (failed(sigOr))
    return failure();

  out.signature = *sigOr;
  return out;
}

llvm::raw_ostream &LIT::operator<<(raw_ostream &os, const MangledSymbol &ms) {
  os << "Mangled: \"";
  // Need to escape the mangled string, it might have some characters that
  // terminals don't like.
  llvm::printEscapedString(ms.mangled.getValue(), os);
  os << "\" - ";
  os << "Modules: [";
  llvm::interleaveComma(ms.moduleNames, os);
  os << "], Structs: [";
  llvm::interleaveComma(ms.structNames, os);
  os << "], Symbol: " << ms.symName;
  os << ", Identifier: " << ms.identifier;
  os << ", Signature: ";
  if (ms.signature)
    os << ms.signature;
  else
    os << "(none)";
  return os;
}

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

void LIT::sortAndDeduplicateTraitSymbols(
    SmallVectorImpl<TraitSymbolAttr> &symbols) {
  llvm::sort(symbols, [&](TraitSymbolAttr ta, TraitSymbolAttr tb) {
    SymbolRefAttr a = ta.getSymbol();
    SymbolRefAttr b = tb.getSymbol();

    if (a.getRootReference() != b.getRootReference())
      return a.getRootReference().getValue() < b.getRootReference().getValue();
    // Compare each segment of the symbols in dictionary order.
    ArrayRef<FlatSymbolRefAttr> aSegments = a.getNestedReferences();
    ArrayRef<FlatSymbolRefAttr> bSegments = b.getNestedReferences();
    for (auto [aSeg, bSeg] : llvm::zip(aSegments, bSegments)) {
      if (aSeg != bSeg)
        return aSeg.getValue() < bSeg.getValue();
    }
    if (aSegments.size() != bSegments.size())
      return aSegments.size() < bSegments.size();

    // Same symbol, then must have the same number of parameter value, compare
    // each parameter lexicographically.
    assert(ta.getParamValues().size() == tb.getParamValues().size());
    for (auto [aVal, bVal] :
         llvm::zip_equal(ta.getParamValues(), tb.getParamValues())) {
      if (aVal != bVal)
        return ParameterAttr::compare(aVal, bVal);
    }
    return false;
  });
  symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
}

std::optional<ParameterEvaluator>
LIT::populateTraitBindingEvaluator(TraitSymbolAttr traitSymbol,
                                   TraitDeclOp traitDecl) {
  // Must be the same trait.
  assert(getFullyResolvedSymbolRef(traitDecl) == traitSymbol.getSymbol());

  // No need to populate the evaluator if the trait symbol has no param values.
  // This should be the common case before we expose parametric trait support to
  // users.
  // NOTE: Normally the signature should have at least one param for Self, but
  // could be empty when the function is called when signature resolving the
  // trait.
  if (traitDecl.getSignature().getParamTypes().size() <= 1)
    return std::nullopt;

  // The decl has an extra Self type parameter.
  assert(traitSymbol.getParamValues().size() + 1 ==
         traitDecl.getSignature().getParamTypes().size());
  ParameterEvaluator evaluator;
  for (size_t i = 0; i < traitSymbol.getParamValues().size(); i++) {
    evaluator.setDeclBinding(traitDecl.getSignature().getParamName(i),
                             traitSymbol.getParamValues()[i]);
  }
  return evaluator;
}

void LIT::canonicalizeTraitCompositionSymbols(
    SmallVectorImpl<TraitSymbolAttr> &symbols,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  // TODO: closure trait need to be deduped differently.
  // Pull in the entire ancestor chain.
  DenseSet<TraitSymbolAttr> seen;
  for (TraitSymbolAttr symbol : symbols) {
    if (!seen.insert(symbol).second)
      continue;

    TraitDeclOp traitOp = traitDeclResolver(symbol.getSymbol());
    auto paramEvaluator = populateTraitBindingEvaluator(symbol, traitOp);

    TraitType canonTrait = traitOp.getCanonicalTrait();
    if (paramEvaluator)
      canonTrait = paramEvaluator->replace(canonTrait);

    // Only one level of parent lookup is needed because parentTypes always
    // include their entire ancestor chain.
    ArrayRef<TraitSymbolAttr> parentSymbols = canonTrait.getSymbols();
    seen.insert(parentSymbols.begin(), parentSymbols.end());
  }
  symbols.assign(seen.begin(), seen.end());

  sortAndDeduplicateTraitSymbols(symbols);
}

FailureOr<TypedAttr> LIT::simplifyConformsToAgainstTypeValue(
    TypeConformsToTraitAttr conformsTo,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  auto traitSymbolsOr = conformsTo.getTraitSymbols();
  if (!traitSymbolsOr)
    return failure();

  TypedAttr typeValues =
      getCanonicalAttr(UpcastAttr::strip(conformsTo.getTypeValue()));
  Type valueType = typeValues.getType();
  // A symbolic pack (e.g. `*Ts`) carries a param-list type; conformance of the
  // whole pack is governed by its element trait bound, so consult the element
  // type. A concrete list is disaggregated into scalar conjuncts upon
  // construction and never reaches here as a list.
  if (auto paramListType = sugarDynCast<ParamListType>(valueType))
    valueType = paramListType.getElementType();
  auto traitType = sugarDynCast<TraitType>(valueType);
  if (!traitType)
    return failure();

  DenseSet<TraitSymbolAttr> symbolSet(traitType.getSymbols().begin(),
                                      traitType.getSymbols().end());
  for (TraitSymbolAttr toCheck : *traitSymbolsOr) {
    if (!symbolSet.contains(toCheck))
      return failure();
  }

  return {SIMDAttr::getScalarBool(conformsTo.getContext(), true)};
}

static LIT::StructType getStructTypeForTypeValue(TypedAttr typeValue) {
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return nullptr;
  return sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
}

TypedAttr LIT::foldDowncastToStructType(DowncastAttr downcast) {
  if (auto structTp = getStructTypeForTypeValue(downcast.getInputTypeValue()))
    // FIXME: We should raise an error when the resolved struct type does not
    // conform to the downcast traits. The folding below is unsafe.
    return TypeParamAttr::get(structTp, downcast.getType());
  return {};
}

FailureOr<ResolvedStructHandle>
LITSymTabEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                            bool acceptAsync) {
  // LITSymTabEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.

  // First try to resolve a LIT struct decl.
  if (auto structType = getStructTypeForTypeValue(typeValue)) {
    if (auto decl = symtab.lookupSymbolIn<StructDeclOp>(
            module, structType.getSymbol())) {
      return ResolvedStructHandle{
          cast<StructDeclInterface>(decl.getOperation()),
          structType.getParamValues(), nullptr,
          /*instance=*/nullptr};
    }
  }
  // Otherwise, fall back to KGEN struct resolution.
  return SymTabEvaluationContext::resolveStructOp(typeValue, acceptAsync);
}

FuncInterface
LITSymTabEvaluationContext::resolveFunctionDecl(SymbolRefAttr symbol) {
  // Functions in the LIT phase are `lit.fn` ops; if not yet lowered to a
  // `kgen.generator`, fall back to the base lookup.
  if (auto fn = symtab.lookupSymbolIn<FnOp>(module, symbol))
    return fn;
  return SymTabEvaluationContext::resolveFunctionDecl(symbol);
}

FailureOr<TypedAttr> LITSymTabEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle TypeConformsToTraitAttr.
  if (auto conformsTo =
          sugarDynCastIfPresent<TypeConformsToTraitAttr>(typedAttr)) {
    auto traitDeclResolver = [&](SymbolRefAttr symbol) -> TraitDeclOp {
      return symtab.lookupSymbolIn<TraitDeclOp>(module, symbol);
    };

    // Try LIT-specific trait type folding first, then fall back to
    // constraint-aware struct resolution.
    FailureOr<TypedAttr> result =
        simplifyConformsToAgainstTypeValue(conformsTo, traitDeclResolver);
    if (succeeded(result))
      return result;

    return conformsTo.evaluateWithContext(*this);
  }

  // Handle DowncastAttr.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (auto structTp =
            getStructTypeForTypeValue(downcast.getInputTypeValue())) {
      // FIXME: We should raise an error when the resolved struct type does not
      // conforms to the downcast traits. The folding below is unsafe.
      return TypeParamAttr::get(structTp, downcast.getType());
    }

    auto toTrait = sugarDynCast<TraitType>(downcast.getType());

    // Extract source trait from the input type value.
    Type fromType = downcast.getInputTypeValue().getType();
    TraitType fromTrait = sugarDynCast<TraitType>(fromType);
    if (auto paramTrait = sugarDynCast<ParamType>(fromType);
        !fromTrait && paramTrait) {
      // if this is a !param<:anytrait<trait>, trait_val>, we can still get a
      // loosest bound from the trait meta type.
      if (auto anyTrait =
              sugarDynCast<AnyTraitType>(paramTrait.getParam().getType())) {
        fromTrait = anyTrait.getTraitType();
      }
      return failure();
    }

    if (!toTrait || !fromTrait)
      return failure();

    llvm::SmallDenseSet<TraitSymbolAttr, 16> fromSymbols(
        fromTrait.getSymbols().begin(), fromTrait.getSymbols().end());
    bool fromImpliesTo =
        llvm::all_of(toTrait.getSymbols(), [&](TraitSymbolAttr symbol) {
          return fromSymbols.contains(symbol);
        });
    if (fromImpliesTo) {
      // If we are downcasting a more-refined trait to a less-refined trait,
      // this is actually an upcast.
      return UpcastAttr::get(downcast.getType(), downcast.getInputTypeValue());
    } else {
      SmallVector<TraitSymbolAttr> allTraitSymbols(fromTrait.getSymbols());
      llvm::append_range(allTraitSymbols, toTrait.getSymbols());
      sortAndDeduplicateTraitSymbols(allTraitSymbols);

      auto allTraits = TraitType::get(attr.getContext(), allTraitSymbols, {});

      auto ret = UpcastAttr::get(
          downcast.getType(),
          DowncastAttr::get(allTraits, downcast.getInputTypeValue()));

      return ret;
    }
  }

  // Delegate to parent class for other context-specific handling.
  return SymTabEvaluationContext::evaluateContextSpecific(attr);
}

//===----------------------------------------------------------------------===//
// IndexToDeclRefRemapper
//===----------------------------------------------------------------------===//

Attribute IndexToDeclRefRemapper::tryReplace(Attribute attr, size_t depth) {
  if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
    if (ref.getDepth() == depth) {
      return ParamDeclRefAttr::get(paramListAttr.getName(ref.getIndex()),
                                   replace(ref.getType()));
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TraitSelfBinder
//===----------------------------------------------------------------------===//

Attribute TraitSelfBinder::tryReplace(Attribute attr, size_t depth) {
  // Replace a reference to $(0,0) with the new selfValue.
  auto paramRef = dyn_cast<ParamIndexRefAttr>(attr);
  if (!paramRef || paramRef.getIndex() != 0 ||
      // Check to see if this is a param ref referring to our Self or some
      // other Self (perhaps in a signature parameter-value that declares its
      // own self or something), see PSTIAIRAID.
      paramRef.getDepth() + 1 != depth)
    return {};
  return selfValue;
}

FnTypeGeneratorType
TraitSelfBinder::bindTraitFnSignature(FnTypeGeneratorType sig) {
  SmallVector<Type> inputTypes(sig.getInputParamTypes());
  inputTypes[0] = ParamType::get(selfValue);

  sig = FnTypeGeneratorType::get(inputTypes, sig.getBodyFnType(),
                                 sig.getParamListAttrs());
  return replace(sig);
}

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttrReplacer
//===----------------------------------------------------------------------===//

Attribute ImplicitOriginToNameRefAttrReplacer::tryReplace(Attribute attr,
                                                          size_t depth) {
  auto implicitOriginRef = dyn_cast<ImplicitOriginRefAttr>(attr);
  if (!implicitOriginRef || implicitOriginRef.getDepth() != depth)
    return nullptr;

  auto it = implicitOriginToNewParamRef.find(implicitOriginRef);
  if (it != implicitOriginToNewParamRef.end())
    return it->second;

  auto originName = StringAttr::get(ctx, llvm::utostr(originDecls.size()) +
                                             "_unnamed`" + namePostfix);
  originDecls.push_back(
      ParamDeclAttr::get(originName, implicitOriginRef.getType()));
  auto originParamRef =
      ParamDeclRefAttr::get(originName, implicitOriginRef.getType());
  implicitOriginToNewParamRef.insert({implicitOriginRef, originParamRef});
  return originParamRef;
}

//===----------------------------------------------------------------------===//
// OriginDeclRemapper
//===----------------------------------------------------------------------===//

NameToImplicitOriginRefRemapper::NameToImplicitOriginRefRemapper(
    ArrayRef<ParamDeclAttr> originDecls, size_t depthOffset)
    : depthOffset(depthOffset) {
  for (auto [index, decl] : llvm::enumerate(originDecls))
    mapping.try_emplace(decl.getName().strref(), index);
}

Attribute NameToImplicitOriginRefRemapper::tryReplace(Attribute attr,
                                                      size_t depth) {
  auto ref = dyn_cast<ParamDeclRefAttr>(attr);
  if (!ref)
    return nullptr;
  // If it's in the mapping, then we know it's an *origin* param ref, so no
  // need to check its type.
  auto it = mapping.find(ref.getName());
  if (it == mapping.end())
    return nullptr;
  return ImplicitOriginRefAttr::get(depth - depthOffset, it->second,
                                    ref.getType());
}

//===----------------------------------------------------------------------===//
// Constraint Implication
//===----------------------------------------------------------------------===//

/// Normalize a `conforms_to` for structural comparison: strip identity wrappers
/// (rebind/upcast/downcast) from the type value so `conforms_to(upcast<T>, X)`
/// matches `conforms_to(T, X)`, and decompose concrete variadic lists and
/// multi-trait checks into an AND of scalar, single-trait `conforms_to`
/// propositions. Returns null if \p prop is not a `conforms_to`, or if it is
/// already normalized and cannot be decomposed.
static TypedAttr decomposeConformsTo(TypedAttr prop) {
  auto conformsTo = dyn_cast<TypeConformsToTraitAttr>(prop);
  if (!conformsTo)
    return {};

  TypedAttr typeValue = stripIdentityWrappers(conformsTo.getTypeValue());

  std::optional<ArrayRef<TraitSymbolAttr>> traitSymbolsOr =
      conformsTo.getTraitSymbols();
  // Not yet resolved.
  if (!traitSymbolsOr)
    return {};

  ArrayRef<TraitSymbolAttr> traitSymbols = *traitSymbolsOr;
  auto concreteList = sugarDynCast<ParamListAttr>(typeValue);
  bool hasMultipleElements =
      concreteList && concreteList.getValues().size() > 1;
  assert(!traitSymbols.empty());
  if (traitSymbols.size() == 1 && !hasMultipleElements) {
    // Nothing to split. Only rebuild when stripping changed the type value;
    // otherwise report "not decomposable" so callers use the prop unchanged.
    if (typeValue == conformsTo.getTypeValue())
      return {};
    return TypeConformsToTraitAttr::get(typeValue, conformsTo.getTraitType());
  }
  // An empty concrete pack is vacuously true; callers handle that via the
  // earlier simplifier paths. Guard here so we never feed an empty operand
  // list to `ParamOperatorAttr::get(POC::And, ...)` (which asserts).
  if (concreteList && concreteList.getValues().empty())
    return {};

  SmallVector<TypedAttr> operands;
  if (concreteList) {
    operands.reserve(concreteList.getValues().size() * traitSymbols.size());
    for (TypedAttr element : concreteList.getValues()) {
      element = stripIdentityWrappers(element);
      for (TraitSymbolAttr sym : traitSymbols) {
        auto singleTrait = TraitType::get(conformsTo.getContext(), {sym});
        operands.push_back(
            TypeConformsToTraitAttr::get(element, singleTrait.getPValue()));
      }
    }
  } else {
    operands.reserve(traitSymbols.size());
    for (TraitSymbolAttr sym : traitSymbols) {
      auto singleTrait = TraitType::get(conformsTo.getContext(), {sym});
      operands.push_back(
          TypeConformsToTraitAttr::get(typeValue, singleTrait.getPValue()));
    }
  }
  return ParamOperatorAttr::get(POC::And, operands);
}

/// Check if prop is NOT(inner), i.e., XOR(inner, true). Returns inner if so.
static TypedAttr getNotOperand(TypedAttr prop) {
  auto xorOp = dyn_cast<ParamOperatorAttr>(prop);
  if (!xorOp || xorOp.getOpcode() != POC::Xor ||
      xorOp.getOperands().size() != 2)
    return {};

  // NOT is represented as XOR(x, true). Check both operand orderings.
  for (auto [maybeInner, maybeTrue] :
       {std::pair{xorOp.getOperand(0), xorOp.getOperand(1)},
        std::pair{xorOp.getOperand(1), xorOp.getOperand(0)}}) {
    if (isTriviallyTrueProposition(maybeTrue))
      return maybeInner;
  }
  return {};
}

/// Ingest the top-level identity facts of `assumption` (and of its AND
/// conjuncts).
static void addEqualityFacts(TypedAttr assumption,
                             llvm::EquivalenceClasses<TypedAttr> &classes) {
  if (std::optional<std::pair<TypedAttr, TypedAttr>> identity =
          getIdentityProposition(assumption)) {
    classes.unionSets(
        stripIdentityWrappers(getCanonicalAttr(identity->first)),
        stripIdentityWrappers(getCanonicalAttr(identity->second)));
  } else if (auto op = dyn_cast<ParamOperatorAttr>(assumption)) {
    if (op.getOpcode() == POC::And)
      for (TypedAttr operand : op.getOperands())
        addEqualityFacts(operand, classes);
  }
}

namespace {
/// The identity facts of an assumption, closed under symmetry & transitivity.
class AssumptionEqualities {
  llvm::EquivalenceClasses<TypedAttr> classes;
  TypedAttr builtFor;

public:
  /// True when `eq(lhs, rhs)` follows from the `eq` facts of `assumption`.
  bool provesEqual(TypedAttr lhs, TypedAttr rhs, TypedAttr assumption) {
    if (builtFor != assumption) {
      classes = llvm::EquivalenceClasses<TypedAttr>();
      addEqualityFacts(assumption, classes);
      builtFor = assumption;
    }
    return classes.isEquivalent(stripIdentityWrappers(getCanonicalAttr(lhs)),
                                stripIdentityWrappers(getCanonicalAttr(rhs)));
  }
};
} // namespace

static TriBool isPropositionImplied(TypedAttr proposition, TypedAttr assumption,
                                    AssumptionEqualities &assumptionEqs) {
  // Canonicalize and decompose multi-trait conforms_to into AND of single-trait
  // ones so the general conjunction rules handle subsumption uniformly.
  proposition = getCanonicalAttr(proposition);
  assumption = getCanonicalAttr(assumption);
  if (TypedAttr decomposed = decomposeConformsTo(proposition))
    proposition = decomposed;
  if (TypedAttr decomposed = decomposeConformsTo(assumption))
    assumption = decomposed;

  // Direct equality: A implies A.
  if (assumption == proposition)
    return TriBool::yes();

  // A trivially false assumption implies anything.
  if (isTriviallyFalseProposition(assumption))
    return TriBool::yes();

  // Trivially true is implied by anything.
  if (isTriviallyTrueProposition(proposition))
    return TriBool::yes();
  // Trivially false constraints are violated under any assumption. This is
  // sound because we know the assumption is not also trivially false here.
  if (isTriviallyFalseProposition(proposition))
    return TriBool::no();

  if (auto assumptionConformance =
          dyn_cast<TypeConformsToTraitAttr>(assumption)) {
    if (auto propositionConformance =
            dyn_cast<TypeConformsToTraitAttr>(proposition)) {
      std::optional<ArrayRef<TraitSymbolAttr>> symbolsA =
          assumptionConformance.getTraitSymbols();
      std::optional<ArrayRef<TraitSymbolAttr>> symbolsB =
          propositionConformance.getTraitSymbols();
      bool traitsImply = false;
      if (symbolsA && symbolsB) {
        DenseSet<TraitSymbolAttr> symbols(symbolsA->begin(), symbolsA->end());
        traitsImply = llvm::all_of(*symbolsB, [&](TraitSymbolAttr symbol) {
          return symbols.contains(symbol);
        });
      }
      if (traitsImply &&
          isEqualCanon(stripIdentityWrappers(getCanonicalAttr(
                           assumptionConformance.getTypeValue())),
                       stripIdentityWrappers(getCanonicalAttr(
                           propositionConformance.getTypeValue()))))
        return TriBool::yes();
    }
  }

  if (std::optional<std::pair<TypedAttr, TypedAttr>> identity =
          getIdentityProposition(proposition)) {
    if (assumptionEqs.provesEqual(identity->first, identity->second,
                                  assumption))
      return TriBool::yes();
  }

  // Conjunction elimination: (A AND B) implies B if any conjunct implies B.
  // AND decomposition: (A AND B) contradicts Z if any conjunct contradicts Z.
  //
  // Scan every conjunct instead of stopping at the first verdict, preferring a
  // proof over a disproof. This must stay ahead of the negation rule below.
  if (auto assumptionOp = dyn_cast<ParamOperatorAttr>(assumption)) {
    if (assumptionOp.getOpcode() == POC::And) {
      bool anyDisproves = false;
      for (Attribute operand : assumptionOp.getOperands()) {
        TriBool result =
            LIT::isPropositionImplied(proposition, cast<TypedAttr>(operand));
        if (result.isFalse())
          anyDisproves = true;
        else if (result.isTrue())
          return TriBool::yes();
      }
      if (anyDisproves)
        return TriBool::no();
    }
  }

  // Negation rule: A contradicts NOT(A).
  // If B = NOT(inner) and A implies inner, then A contradicts B.
  if (TypedAttr innerProposition = getNotOperand(proposition))
    if (isPropositionImplied(innerProposition, assumption, assumptionEqs)
            .isTrue())
      return TriBool::no();
  // Symmetric: if A = NOT(inner) and B implies inner, then A contradicts B.
  if (TypedAttr innerAssumption = getNotOperand(assumption))
    if (isImplicationProven(innerAssumption, proposition))
      return TriBool::no();

  if (auto propositionOp = dyn_cast<ParamOperatorAttr>(proposition)) {
    // Weakening: A implies (A OR B) for any B.
    if (propositionOp.getOpcode() == POC::Or) {
      for (Attribute operand : propositionOp.getOperands())
        if (isPropositionImplied(cast<TypedAttr>(operand), assumption,
                                 assumptionEqs)
                .isTrue())
          return TriBool::yes();
    }
    // Conjunction introduction: A implies (B AND C) iff A implies every
    // conjunct. A contradicts (B AND C) if A contradicts any conjunct.
    if (propositionOp.getOpcode() == POC::And) {
      TriBool result = TriBool::yes();
      for (Attribute operand : propositionOp.getOperands()) {
        TriBool operandResult = isPropositionImplied(cast<TypedAttr>(operand),
                                                     assumption, assumptionEqs);
        if (operandResult.isFalse())
          return TriBool::no();
        if (operandResult.isUnknown())
          result = TriBool::unknown();
      }
      return result;
    }
  }

  // Fallback: A implies B iff AND(A, B) == A.
  TypedAttr combined =
      ParamOperatorAttr::get(POC::And, {assumption, proposition});
  if (combined == assumption)
    return TriBool::yes();

  return TriBool::unknown();
}

TriBool LIT::isPropositionImplied(TypedAttr proposition,
                                  ArrayRef<TypedAttr> assumptions) {
  TypedAttr combinedAssumption;
  if (assumptions.empty())
    combinedAssumption =
        SIMDAttr::getScalarBool(proposition.getContext(), true);
  else if (assumptions.size() == 1)
    combinedAssumption = assumptions.front();
  else
    combinedAssumption = ParamOperatorAttr::get(POC::And, assumptions);

  return isPropositionImplied(proposition, combinedAssumption);
}

TriBool LIT::isPropositionImplied(ConstraintAttr proposition,
                                  ArrayRef<ConstraintAttr> assumptions,
                                  ParameterEvaluator &evaluator) {
  TypedAttr propositionAttr = getCanonicalAttr(proposition.getProposition());
  TypedAttr reboundProposition =
      getCanonicalAttr(evaluator.getReboundAttribute(propositionAttr));
  if (isTriviallyTrueProposition(reboundProposition))
    return TriBool::yes();

  SmallVector<TypedAttr> canonAssumptions;
  canonAssumptions.reserve(assumptions.size());
  for (ConstraintAttr assumption : assumptions)
    canonAssumptions.push_back(getCanonicalAttr(assumption.getProposition()));

  return isPropositionImplied(reboundProposition, canonAssumptions);
}

TriBool LIT::isPropositionImplied(TypedAttr proposition, TypedAttr assumption) {
  AssumptionEqualities assumptionEqs;
  return ::isPropositionImplied(proposition, assumption, assumptionEqs);
}

/// Visit each TypeConformsToTraitAttr found in a constraint proposition.
/// Canonical AND is already flattened to a single n-ary node, so a single
/// top-level loop over its operands is sufficient. OR / NOT are not visited
/// since they are not definite knowledge.
static void forEachConformsToInProposition(
    TypedAttr proposition,
    llvm::function_ref<void(TypeConformsToTraitAttr)> callback) {
  proposition = getCanonicalAttr(proposition);

  auto visit = [&](TypedAttr attr) {
    if (auto ct = dyn_cast<TypeConformsToTraitAttr>(getCanonicalAttr(attr)))
      callback(ct);
  };

  // Canonical AND is flattened to a single n-ary node, so iterate its
  // operands directly. Otherwise treat the proposition itself as a single
  // candidate.
  if (auto op = dyn_cast<ParamOperatorAttr>(proposition);
      op && op.getOpcode() == POC::And) {
    for (TypedAttr operand : op.getOperands())
      visit(operand);
    return;
  }
  visit(proposition);
}

TraitType LIT::getTraitBoundFromAssumptions(
    TypedAttr typeAttr, ArrayRef<ConstraintAttr> assumptions,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  typeAttr = getCanonicalAttr(typeAttr);

  if (assumptions.empty())
    return {};

  TypedAttr targetStripped = stripIdentityWrappers(typeAttr);

  auto targetParamListGet = dyn_cast<ParamListGetAttr>(targetStripped);
  TypedAttr targetParamList;
  if (targetParamListGet)
    targetParamList = stripIdentityWrappers(
        getCanonicalAttr(targetParamListGet.getParamList()));

  // Collect trait symbols from all relevant conforms_to constraints.
  SmallVector<TraitSymbolAttr> allTraits;
  for (ConstraintAttr assumption : assumptions) {
    forEachConformsToInProposition(
        assumption.getProposition(), [&](TypeConformsToTraitAttr ct) {
          TypedAttr striped =
              stripIdentityWrappers(getCanonicalAttr(ct.getTypeValue()));
          if (isEqualCanon(striped, targetStripped) ||
              (targetParamList && isEqualCanon(striped, targetParamList))) {
            std::optional<ArrayRef<TraitSymbolAttr>> symOr =
                ct.getTraitSymbols();
            if (!symOr)
              return;
            allTraits.append(symOr->begin(), symOr->end());
          }
        });
  }

  if (allTraits.empty())
    return {};

  // Canonicalize to include ancestor traits.
  sortAndDeduplicateTraitSymbols(allTraits);

  return TraitType::get(typeAttr.getContext(), allTraits);
}

ParamDeclRefAttr LIT::extractParamDeclRef(TypedAttr attr) {
  if (auto upcast = dyn_cast<UpcastAttr>(attr))
    return extractParamDeclRef(upcast.getInputTypeValue());

  if (auto typeParam = dyn_cast<TypeParamAttr>(attr)) {
    Type innerType = typeParam.getTypeValue();
    if (auto innerParamType = dyn_cast<ParamType>(innerType))
      return extractParamDeclRef(innerParamType.getParam());
  }

  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr))
    return paramRef;

  return {};
}
