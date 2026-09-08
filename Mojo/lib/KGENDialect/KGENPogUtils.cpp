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
// This file implements POG (parameter/argument metadata) related parsing,
// printing, and verification helpers in the KGEN namespace.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "llvm/Support/SMLoc.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Parameter Mangling
//===----------------------------------------------------------------------===//

/// Demangle a mangled parameter name if it is has a "`" postfix and and
/// trailing depth and unique ID.
StringRef KGEN::demangleParameterName(StringRef name, bool forUser) {
  if (name.empty())
    return {};
  // Remove any uniquing number suffix.
  name = name.take_front(name.find('`'));
  // Remove any type prefixes if present.
  if (forUser) {
    auto dotLoc = name.find_last_of('.');
    if (dotLoc != StringRef::npos) {
      name = name.drop_front(dotLoc + 1);
      assert(!name.empty());
    }
  }
  return name;
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

ParseResult KGEN::parseOptionalDefaultValue(AsmParser &p, TypedAttr &defaultVal,
                                            Type type) {
  if (succeeded(p.parseOptionalEqual())) {
    // The default value can mismatch; if so, it will have a :type prefix that
    // overrides the expected type.
    (void)parseColonTypeOrDefault(p, type, type);
    return parseParamValue(p, defaultVal, type);
  }
  return success();
}

void KGEN::printOptionalDefaultValue(AsmPrinter &p, TypedAttr defaultVal,
                                     Type type) {
  if (!defaultVal)
    return;

  p << " = ";
  // The default value can mismatch the expected type; if so, print the
  // actual type.
  if (defaultVal.getType() != type) {
    p << ':';
    printKGENType(p, defaultVal.getType());
    p << ' ';
  }
  printParamValue(p, defaultVal);
}

/// Helper to parse sigils that indicate that an argument/parameter is variadic
/// or a pack. The given index is emplaced in the appropriate list of indices,
/// if a `var` or `pack` sigil is parsed.
static ParseResult parseVariadicness(AsmParser &p, VariadicKind &variadic,
                                     size_t idx) {
  mlir::SMLoc loc = p.getCurrentLocation();
  StringRef sigil;
  if (succeeded(p.parseOptionalKeyword(&sigil))) {
    if (std::optional<VariadicKind> kind = symbolizeVariadicKind(sigil)) {
      variadic = *kind;
    } else {
      return p.emitError(loc, "expected variadic kind, got: ") << sigil;
    }
  }
  return success();
}

/// Parse a parameter spec if present, including input parameter declarations,
/// and default values.
/// parameter-decl   ::= identifier (`[` identifier `]`)?
///                        (`:` type (`=` expression)? )?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` (parameter-list (`,` `{` constraint-list `}`)?
///                        | `{` constraint-list `}`)? `>`
static ParseResult
parseConstraintsListContents(AsmParser &p,
                             SmallVectorImpl<ConstraintAttr> &constraints) {
  do {
    auto constraint =
        llvm::cast_if_present<ConstraintAttr>(ConstraintAttr::parse(p, {}));
    if (!constraint)
      return failure();
    constraints.push_back(cast<ConstraintAttr>(constraint));
  } while (succeeded(p.parseOptionalComma()));
  if (p.parseRBrace())
    return failure();
  return success();
}

ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &p,
                                 ParamDeclArrayAttr &inputParamDecls,
                                 PogListAttr &paramListAttr) {
  MLIRContext *ctx = p.getContext();
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> defaultParams;
  SmallVector<VariadicKind> argsVariadic;
  std::optional<ArgConvention> origVariadicConvention;
  SmallVector<ConstraintAttr> bodyConstraints;
  bool parsedBodyConstraints = false;

  llvm::SMLoc startLoc = p.getCurrentLocation();
  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseWithDefault =
      [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    if (!parsedBodyConstraints && succeeded(p.parseOptionalLBrace())) {
      if (failed(parseConstraintsListContents(p, bodyConstraints)))
        return failure();
      parsedBodyConstraints = true;
      return success();
    }
    if (parsedBodyConstraints)
      return p.emitError(p.getCurrentLocation(),
                         "expected '>' after body constraints");

    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    StringAttr paramName;
    if (succeeded(p.parseOptionalLSquare())) {
      std::string str;
      if (p.parseString(&str) || p.parseRSquare())
        return failure();
      paramName = StringAttr::get(ctx, str);
    }

    ParamDeclAttr decl;
    if (failed(parseParamDecl(p, decl)))
      return failure();
    decls.emplace_back(decl);

    // We store an empty string for the name of implicit parameters.
    bool isImplicit = passingKindParser.isCurrentImplicit();
    if (!paramName) {
      paramName = StringAttr::get(
          ctx, isImplicit ? "" : demangleParameterName(decl.getName()));
    }
    paramNames.emplace_back(paramName);

    if (failed(parseVariadicness(
            p, argsVariadic.emplace_back(VariadicKind::None), idx++)))
      return failure();

    // Parameters don't have ArgConvention's.
    if (argsVariadic.back() == VariadicKind::PackVarArg ||
        argsVariadic.back() == VariadicKind::PosVarArg)
      origVariadicConvention = ArgConvention::ImmReg;

    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, decl.getType())))
      return failure();
    defaultParams.push_back(defaultVal);

    return success();
  };

  ParamDeclArrayAttr resultParamDecls;
  if (failed(KGEN::parseOptionalParameterSpec(
          p, inputParamDecls, resultParamDecls, parseWithDefault)))
    return failure();
  if (!resultParamDecls.empty())
    return p.emitError(startLoc, "expected no result parameters");

  passingKindParser.populatePassingKinds(paramPassingKinds);

  paramListAttr =
      PogListAttr::get(ctx, paramNames, paramPassingKinds, argsVariadic,
                       defaultParams, origVariadicConvention, bodyConstraints);
  return success();
}

static void printConstraintsListContents(AsmPrinter &p,
                                         ArrayRef<ConstraintAttr> constraints,
                                         ParameterEvaluator *evaluator) {
  llvm::interleaveComma(constraints, p, [&](ConstraintAttr constraint) {
    if (evaluator)
      constraint =
          cast<ConstraintAttr>(evaluator->getReboundAttribute(constraint));
    constraint.print(p);
  });
}

ParseResult KGEN::parseConventionAndVariadicness(
    AsmParser &p, ArgConvention &convention, VariadicKind &variadic,
    std::optional<ArgConvention> &origVariadicConvention, size_t idx) {
  mlir::SMLoc loc = p.getCurrentLocation();
  StringRef str;
  convention = ArgConvention::ImmReg;
  if (succeeded(p.parseOptionalKeyword(&str))) {
    if (std::optional<ArgConvention> conv = symbolizeArgConvention(str)) {
      convention = *conv;
      // If we just had a convention and no vertical bar, we're done.
      if (failed(p.parseOptionalVerticalBar()))
        return success();
      // Otherwise we also parse a variadicness
      if (parseVariadicness(p, variadic, idx))
        return failure();
      if (variadic == VariadicKind::PosVarArg ||
          variadic == VariadicKind::PackVarArg) {
        origVariadicConvention = convention;
        if (convention != ArgConvention::OwnedMem)
          convention = ArgConvention::ImmMem;
      }
      return success();
    }

    std::optional<VariadicKind> kind = symbolizeVariadicKind(str);
    if (!kind.has_value())
      return p.emitError(loc, "expected convention|variadicness, got: ") << str;
    variadic = *kind;

    if (variadic == VariadicKind::PosVarArg ||
        variadic == VariadicKind::PackVarArg) {
      origVariadicConvention = convention;
      if (convention != ArgConvention::OwnedMem)
        convention = ArgConvention::ImmMem;
    }
  }
  return success();
}

/// Print the variadicness as a strings.
static void printVariadicness(AsmPrinter &p, VariadicKind variadicness,
                              char separator = ' ') {
  if (variadicness != VariadicKind::None)
    p << separator << stringifyVariadicKind(variadicness);
}

void KGEN::printConventionAndVariadicness(AsmPrinter &p,
                                          ArgConvention convention,
                                          VariadicKind variadicness) {
  if (convention != ArgConvention::ImmReg) {
    p << ' ' << stringifyArgConvention(convention);
    printVariadicness(p, variadicness, '|');
  } else if (variadicness != VariadicKind::None) {
    p << ' ' << stringifyVariadicKind(variadicness);
  }
}

void KGEN::printOptionalParameterSpec(AsmPrinter &p,
                                      ArrayRef<ParamDeclAttr> paramDecls,
                                      PogListAttr paramListAttr,
                                      ParameterEvaluator &evaluator) {
  // Substitute input parameters when printing default parameters.
  for (ParamDeclAttr param : paramDecls)
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(param));

  size_t idx = 0;
  PassingKindPrinter passingKindPrinter(p, paramListAttr, '|');
  ArrayRef<ConstraintAttr> bodyConstraints = paramListAttr.getBodyConstraints();
  auto printWithDefault = [&](ParamDeclAttr decl) {
    passingKindPrinter.printOptionalStarSlash(idx);

    StringAttr name = paramListAttr.getName(idx);
    // If we can't encode the parameter name inside the mangled decl name, then
    // print it explicitly.
    if (paramListAttr.getPassingKind(idx) != PassingKind::Implicit &&
        name != demangleParameterName(decl.getName()))
      p << '[' << name << ']';
    printParamDecl(p, decl);
    printVariadicness(p, paramListAttr.getVariadicKind(idx));

    if (TypedAttr defaultOr = paramListAttr.getDefault(idx)) {
      printOptionalDefaultValue(p, evaluator.getReboundAttribute(defaultOr),
                                decl.getType());
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
    if (idx == paramDecls.size() && !bodyConstraints.empty()) {
      p << ", {";
      printConstraintsListContents(p, bodyConstraints, &evaluator);
      p << '}';
    }
  };
  if (paramDecls.empty() && bodyConstraints.empty())
    return;
  if (!paramDecls.empty()) {
    KGEN::printOptionalParameterSpec(p, paramDecls, /*resultParams=*/{},
                                     printWithDefault);
    return;
  }

  p << "<{";
  printConstraintsListContents(p, bodyConstraints, &evaluator);
  p << "}>";
}

ParseResult KGEN::parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    PogListAttr &paramListAttr, function_ref<ParseResult()> parseBody) {
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> defaultParams;
  SmallVector<VariadicKind> argVariadics;
  std::optional<ArgConvention> origVariadicConvention;
  SmallVector<ConstraintAttr> bodyConstraints;
  bool parsedBodyConstraints = false;

  // Parse the input parameter types and optional default values.
  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseInputParam = [&](SmallVectorImpl<Type> &inputs) -> ParseResult {
    if (!parsedBodyConstraints && succeeded(p.parseOptionalLBrace())) {
      if (failed(parseConstraintsListContents(p, bodyConstraints)))
        return failure();
      parsedBodyConstraints = true;
      return success();
    }
    if (parsedBodyConstraints)
      return p.emitError(p.getCurrentLocation(),
                         "expected '>' after body constraints");

    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    // Parse an optional parameter name.
    if (parseOptionalName(p, paramNames.emplace_back()))
      return {};

    Type &type = inputs.emplace_back();
    if (failed(parseKGENType(p, type)))
      return failure();

    if (failed(parseVariadicness(
            p, argVariadics.emplace_back(VariadicKind::None), idx++)))
      return failure();

    if (argVariadics.back() == VariadicKind::PackVarArg ||
        argVariadics.back() == VariadicKind::PosVarArg)
      origVariadicConvention = ArgConvention::ImmMem;

    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, type)))
      return failure();
    defaultParams.push_back(defaultVal);

    return success();
  };

  if (failed(KGEN::parseOptionalParamSignature(p, inputParamTypes,
                                               parseInputParam)))
    return failure();

  passingKindParser.populatePassingKinds(paramPassingKinds);

  if (parseBody && failed(parseBody()))
    return failure();

  paramListAttr = PogListAttr::get(
      p.getContext(), paramNames, paramPassingKinds, argVariadics,
      defaultParams, origVariadicConvention, bodyConstraints);
  return success();
}

void KGEN::printOptionalParamSignature(AsmPrinter &p,
                                       ArrayRef<Type> inputParamTypes,
                                       PogListAttr paramListAttr,
                                       bool omitEmptyAngleBrackets) {
  ArrayRef<ConstraintAttr> bodyConstraints = paramListAttr.getBodyConstraints();
  if (inputParamTypes.empty()) {
    if (!bodyConstraints.empty()) {
      p << "<{";
      printConstraintsListContents(p, bodyConstraints, /*evaluator=*/nullptr);
      p << "}>";
      return;
    }
    if (!omitEmptyAngleBrackets)
      p << "<>";
    return;
  }

  size_t idx = 0;
  PassingKindPrinter passingKindPrinter(p, paramListAttr, '|');
  auto printWithDefault = [&](Type type) {
    passingKindPrinter.printOptionalStarSlash(idx);

    if (StringAttr name = paramListAttr.getName(idx); !name.empty()) {
      p.printString(name);
      p << ": ";
    }
    printKGENType(p, type);
    printVariadicness(p, paramListAttr.getVariadicKind(idx));
    printOptionalDefaultValue(p, paramListAttr.getDefault(idx), type);

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
    if (idx == inputParamTypes.size() && !bodyConstraints.empty()) {
      p << ", {";
      printConstraintsListContents(p, bodyConstraints, /*evaluator=*/nullptr);
      p << '}';
    }
  };

  KGEN::printOptionalParamSignature(p, inputParamTypes, printWithDefault);
}

ParseResult KGEN::parseOptionalName(AsmParser &p, StringAttr &name) {
  std::string argName;
  if (succeeded(p.parseOptionalString(&argName)))
    if (failed(p.parseColon()))
      return failure();
  name = StringAttr::get(p.getContext(), argName);
  return success();
}

//===----------------------------------------------------------------------===//
// PassingKindParser / PassingKindPrinter
//===----------------------------------------------------------------------===//

static std::optional<PassingKindParser::Marker>
parseOptionalMarker(AsmParser &p) {
  // We want to allow a standalone * in the signature to represent information
  // about a signature list, but we don't want to interfere with *"fo o" escaped
  // name parsing.  Do a bit of grotty lookahead to make sure we're ok to
  // consume a star.  While grotty, this cannot overrun the end of the file
  // because the MLIR asmparser guarantees the buffer is always NUL terminated.
  // This doesn't support whitespace/comments etc between the star and quote
  // though.
  llvm::SMLoc loc = p.getCurrentLocation();
  if (loc.getPointer()[0] == '*' && loc.getPointer()[1] == '"')
    return {};

  if (succeeded(p.parseOptionalPlus()))
    return PassingKindParser::PLUS;
  if (succeeded(p.parseOptionalVerticalBar()))
    return PassingKindParser::BAR;
  if (succeeded(p.parseOptionalStar()))
    return PassingKindParser::STAR;
  if (succeeded(p.parseOptionalQuestion()))
    return PassingKindParser::QUESTION;
  return {};
}

OptionalParseResult PassingKindParser::parseOptionalStarSlash() {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::optional<Marker> marker = parseOptionalMarker(parser);
  if (!marker) {
    ++idx;
    return std::nullopt;
  }

  // Error if the same marker was already found.
  if (foundMarkers[*marker]) {
    return parser.emitError(loc, "only one '")
           << markers[*marker] << "' allowed in signature";
  }
  // Error if any markers that are supposed to come after were already parsed.
  for (int i = *marker + 1; i < NUM_MARKERS; ++i) {
    if (foundMarkers[i]) {
      return parser.emitError(loc, "'") << markers[i] << "' cannot precede '"
                                        << markers[*marker] << "' in signature";
    }
  }

  foundMarkers[*marker] = true;
  idxOfEach[*marker] = idx;
  return mlir::success();
}

void PassingKindParser::populatePassingKinds(
    SmallVectorImpl<PassingKind> &kinds) const {
  size_t lastIdx = 0;
  // Compute the number of elements from the previous marker to this marker.
  std::array<size_t, NUM_MARKERS + 1> fwdSegments{}, revSegments{};
  for (int i = 0; i < NUM_MARKERS; ++i) {
    if (foundMarkers[i]) {
      fwdSegments[i] = idxOfEach[i] - lastIdx;
      lastIdx = idxOfEach[i];
    } else {
      fwdSegments[i] = 0;
    }
  }
  fwdSegments[NUM_MARKERS] = idx - lastIdx;

  // Compute the number of elements from the next marker to this marker.
  lastIdx = idx;
  for (int i = NUM_MARKERS - 1; i >= 0; --i) {
    if (foundMarkers[i]) {
      revSegments[i] = lastIdx - idxOfEach[i];
      lastIdx = idxOfEach[i];
    } else {
      revSegments[i] = 0;
    }
  }
  revSegments[0] = lastIdx;

  // Number of inferred and positional only are the number of elements that come
  // before the marker, until the previous marker or beginning. Number of
  // implicit or keyword-only are the number of elements that come after the
  // marker, until the next marker or the end. The number of keyword or position
  // is everything else.
  kinds.append(fwdSegments[PLUS], PassingKind::Inferred);
  kinds.append(fwdSegments[BAR], PassingKind::PosOnly);
  kinds.append(idx - fwdSegments[PLUS] - fwdSegments[BAR] - revSegments[STAR] -
                   revSegments[QUESTION],
               PassingKind::PosOrKw);
  kinds.append(revSegments[STAR], PassingKind::KwOnly);
  kinds.append(revSegments[QUESTION], PassingKind::Implicit);
}

PassingKindPrinter::PassingKindPrinter(
    raw_ostream &os, size_t numPogs,
    std::function<PassingKind(size_t)> getPassingKind,
    bool suppressSlashAfterSelf, char slash, StringRef plus)
    : os(os), numPogs(numPogs), getPassingKind(std::move(getPassingKind)),
      prevPassingKind(PassingKind::Inferred),
      suppressSlashAfterSelf(suppressSlashAfterSelf), slash(slash), plus(plus) {
}

PassingKindPrinter::PassingKindPrinter(raw_ostream &os, PogListAttr pogListAttr,
                                       bool suppressSlashAfterSelf, char slash,
                                       StringRef plus)
    : PassingKindPrinter(
          os, pogListAttr.size(),
          [pogListAttr](size_t idx) { return pogListAttr.getPassingKind(idx); },
          suppressSlashAfterSelf, slash, plus) {}

PassingKindPrinter::PassingKindPrinter(AsmPrinter &printer,
                                       PogListAttr pogListAttr, char slash,
                                       StringRef plus)
    : PassingKindPrinter(printer.getStream(), pogListAttr,
                         /*suppressSlashAfterSelf=*/false, slash, plus) {}

void PassingKindPrinter::printOptionalStarSlash(size_t idx) {
  // When the pog list is empty (a `FuncType` constructed outside the Mojo
  // parser with no source-level metadata), there are no passing-kind
  // transitions to print.
  if (numPogs == 0)
    return;
  PassingKind passingKind = getPassingKind(idx);
  if (prevPassingKind == passingKind)
    return;

  switch (prevPassingKind) {
  case PassingKind::Inferred:
    if (idx != 0)
      os << plus << ", ";
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::PosOnly:
    // Check if we are in the starting state; if no, this was the last
    // positional-only argument. Optionally, we may want to suppress '/' before
    // the second argument.
    assert(idx != 0);
    if (!suppressSlashAfterSelf || idx != 1)
      os << slash << ", ";
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::PosOrKw:
    assert(passingKind != PassingKind::PosOnly &&
           "positional-only argument cannot follow positional-or-keyword");
    if (passingKind == PassingKind::KwOnly)
      os << "*, ";
    else if (passingKind == PassingKind::Implicit)
      os << "?, ";
    break;
  case PassingKind::KwOnly:
    assert(passingKind == PassingKind::Implicit);
    os << "?, ";
    break;
  case PassingKind::Implicit:
    llvm_unreachable("implicit must be the last passing kind");
  }
  prevPassingKind = passingKind;
}

void PassingKindPrinter::printOptionalTrailingSlash(size_t idx) const {
  if (suppressSlashAfterSelf && idx == 0)
    return;
  if (idx == numPogs - 1) {
    if (prevPassingKind == PassingKind::PosOnly)
      os << ", " << slash;
    else if (prevPassingKind == PassingKind::Inferred)
      os << ", " << plus;
  }
}

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

LogicalResult
KGEN::verifyPassingKinds(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<PogMetadataAttr> pogs, StringRef argOrParam) {
  // First, verify the order of passing kinds.
  auto latestKind = PassingKind::PosOnly;
  auto emitDiag = [&](PassingKind kind) {
    return emitError() << stringifyPassingKind(kind)
                       << " passing kind cannot follow "
                       << stringifyPassingKind(latestKind);
  };

  for (PogMetadataAttr pogAttr : pogs) {
    PassingKind kind = pogAttr.getPassingKind();
    if (kind == PassingKind::Implicit) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::Implicit)
      return emitDiag(kind);
    if (kind == PassingKind::KwOnly) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::KwOnly)
      return emitDiag(kind);
    if (kind == PassingKind::PosOrKw) {
      latestKind = kind;
      continue;
    }
    if (latestKind == PassingKind::PosOrKw)
      return emitDiag(kind);
    assert(latestKind == PassingKind::PosOnly);
  }
  return success();
}
