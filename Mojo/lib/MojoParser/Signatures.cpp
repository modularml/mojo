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
// This contains logic for parsing and type checking and IR building of function
// signatures.  This is used both for fn/def declarations, but also for function
// type syntax.
//
//===----------------------------------------------------------------------===//

#include "Signatures.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamInf.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"

#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

TypedAttr ASTType::extractStructField(TypedAttr value, StringRef fieldName,
                                      SMLoc loc, SharedState &shared) {
  ASTDecl *typeDecl = ASTType(value.getType()).getDecl(shared);
  if (!typeDecl || !sugarIsa<LIT::StructType>(value.getType()))
    return {};

  // Look up the field in the type.
  LookupResult lookup =
      shared.lookupAndResolveDecl(fieldName, loc, *typeDecl,
                                  /*searchParentScopes=*/false);
  if (lookup.getIfSuccess().size() != 1)
    return {};
  auto lookupResult = lookup.getIfSuccess()[0];
  if (!lookupResult)
    return {};
  auto fieldOp =
      dyn_cast_or_null<StructFieldOp>(lookupResult->getIfOperation());
  if (!fieldOp)
    return {};

  return LIT::StructExtractAttr::get(value, fieldOp);
}

/// Given an expression that can be used in `origin_of` or a ref expression,
/// analyze it to determine which origin it represents.  If it doesn't work,
/// emit an error and return null.
TypedAttr IREmitter::extractOriginOf(const ExprNode *expr, CValue value) {
  // Check for !lit.origin and Origin struct.
  if (auto pv = value.getIfPValue()) {
    if (TypedAttr result = ASTType::extractOriginOf(pv.get()))
      return result;
  }

  if (Value ref = emitRefValue({value, expr}, EC_Origin))
    return cast<RefType>(ref.getType()).getOrigin();
  return {};
}

enum class RefSpecifierKind { kRefArgument, kOutArgument, kRefResult };

/// Process the origin expression in a `ref [...] T` or 'out [...] T' reference
/// specifier. This returns the result !lit.ref type.
static RefType processRefOriginSpecifier(const ExprNode *origExpr, ASTType type,
                                         StringRef valueName,
                                         TypeCheckedParamList &paramList,
                                         RefSpecifierKind refKind) {
  SharedState &shared = paramList.shared;

  // For errors, return "RefType(TypeCheckErrorType)" to maintain the invariant
  // that all "ref" values have RefType, but their RValue type is an error.
  auto hadError = [&]() -> RefType {
    return RefType::getAnyOrigin(shared.getTypeCheckErrorType(),
                                 /*isMut*/ true);
  };

  // Propagate already diagnosed errors.
  if (isa<TypeCheckErrorType>(type))
    return hadError();

  IREmitter emitter(paramList.declScope, EC_Origin);

  // Check to see if this is a value address space specifier.  If so, return
  // true, otherwise return false.
  auto digOutAddressSpace = [&](TypedAttr value, SMLoc loc) -> TypedAttr {
    // If the value has index type, then it is good to go.
    if (value.getType().isIndex())
      return value;

    // Check to see if this is the well-known AddressSpace struct.  If so,
    // dig out the index from within it.
    auto extractInt = ASTType::extractStructField(value, "_value", loc, shared);
    if (!extractInt)
      return {};
    auto extractIndex =
        ASTType::extractStructField(extractInt, "_mlir_value", loc, shared);
    if (!extractIndex)
      return {};
    if (extractIndex.getType().isIndex())
      return extractIndex;
    return {};
  };

  // If the origin expression is syntactically a multi-element tuple, then
  // take it apart.
  ArrayRef<const ExprNode *> originExprElts;
  if (auto *tuple = dyn_cast_if_present<TupleNode>(origExpr))
    originExprElts = tuple->exprs;
  else if (origExpr)
    originExprElts = origExpr;

  // Emit the origin expression if it is a normal expression.
  TypedAttr origin;
  TypedAttr addrSpace;
  for (const ExprNode *expr : originExprElts) {
    // Ignore _'s.
    if (expr->kind == ExprNode::kDiscardLiteral)
      continue;

    // The origin expression may be any of:
    //   1) an MValue, which we take the origin from.
    //   2) a value of !lit.origin or Origin[Mut] type.
    //  In the former case, we want to evaluate the expression without
    // evaluating it, because it may involve complex nested expressions and we
    // may be in a PValue expression.
    TypedAttr thisOrigin;
    bool isError = false;
    emitter.emitExpressionWithoutEvaluatingIt(
        expr, EC_Origin, [&](CValue result, IREmitter &emitter) {
          // Check to see if it is an address space first.
          if (auto pv = result.getIfPValue()) {
            if (auto as = digOutAddressSpace(pv.get(), expr->getLoc())) {
              if (addrSpace) {
                emitter.emitError(expr->getLoc())
                    << "address space must be specified once; remove duplicate "
                       "address space specifications"
                    << expr->getRange();
              }
              addrSpace = as;
              return;
            }
          }
          // Otherwise it must be a !lit.origin and Origin struct.
          thisOrigin = emitter.extractOriginOf(expr, result);
          isError = !thisOrigin;
        });

    if (isError)
      return hadError();

    // If we found an origin, add it to our set.
    if (!thisOrigin)
      continue;
    if (!origin)
      origin = thisOrigin;
    else
      origin = OriginUnionAttr::get(origin.getContext(), {origin, thisOrigin});
  }

  // If no origin is specified, then it is inferred from the callsite. Add two
  // parameters to this function: one for the mutability of type Bool and one
  // for the origin.
  if (!origin) {
    auto addParam = [&](const Twine &name, Type type) -> TypedAttr {
      auto paramDecl =
          ParamDeclAttr::get(paramList.declScope.mangleParamName(name), type);
      paramList.names.push_back(paramDecl.getName());
      paramList.passingKinds.push_back(PassingKind::Implicit);
      paramList.paramDeclAttrs.push_back(paramDecl);
      paramList.locations.push_back(origExpr ? origExpr->getLoc() : SMLoc());
      paramList.variadicKinds.push_back(VariadicKind::None);
      paramList.defaults.push_back(TypedAttr());
      return ParamDeclRefAttr::get(paramDecl);
    };

    TypedAttr isMut;
    switch (refKind) {
    case RefSpecifierKind::kRefArgument:
      // "ref [_] arg" infers mutability from the argument.
      isMut = addParam(valueName + "_is_mut",
                       SIMDType::get(shared.getContext(), 1, KGENDType::kBool));
      break;
    case RefSpecifierKind::kOutArgument:
      // "out [_] arg" infers the origin but is always mutable.
      isMut = SIMDAttr::getScalarBool(shared.getContext(), true);
      break;
    case RefSpecifierKind::kRefResult:
      emitter.emitError(origExpr->getLoc())
          << "cannot infer origin for a function result"
          << origExpr->getRange();
      return hadError();
    }
    origin = addParam(valueName + "_is_origin", OriginType::get(isMut));
  }
  if (!origin)
    return hadError();

  if (!isa<OriginType>(origin.getType())) {
    emitter.emitError(origExpr->getLoc())
        << "result reference origin has unexpected type "
        << ASTType(origin.getType()) << origExpr->getRange();
    return hadError();
  }

  if (!addrSpace)
    addrSpace = IntegerAttr::get(IndexType::get(shared.getContext()), 0);

  return RefType::get(type, origin, addrSpace);
}

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

ParseResult ParsedArgument::parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                                  ArgListKind kind) {
  loc = p.getToken().getLoc();
  cursor = p.getLexer().getCursor();

  auto handleContextualArgConvention = [&](StringRef str,
                                           PAArgConvention conv) {
    // Handle "out: Foo" as a name if we're in a parameter list, because
    // they don't get argument conventions.
    if (kind != ArgListKind::kFnTypeParamList &&
        kind != ArgListKind::kParamList) {
      convention = conv;
    } else {
      // Otherwise, the "out" is the argument name.
      name = StringAttr::get(p.getContext(), str);
    }
  };

  // Any var/read/imm/mut/ref keyword sets convention.
  if (p.consumeIf(Token::kw_var)) {
    convention = kConventionVar;
  } else if (p.consumeIf(Token::kw_ref)) {
    (void)p.parseRefSpecifier(refOriginExpr);
    convention = kConventionRef;
  } else if (p.consumeIfSoftIdentifier("out")) {
    handleContextualArgConvention("out", kConventionOut);
    if (convention == kConventionOut)
      (void)p.parseRefSpecifier(refOriginExpr);
  } else if (p.consumeIfSoftIdentifier("mut")) {
    handleContextualArgConvention("mut", kConventionMut);
  } else if (p.consumeIfSoftIdentifier("imm")) {
    handleContextualArgConvention("imm", kConventionImm);
  } else if (p.consumeIfSoftIdentifier("read")) {
    handleContextualArgConvention("read", kConventionImm);
    if (convention == kConventionImm)
      p.emitError(loc, "'read' was removed; use 'imm'")
          << FixIt::replaceToken(loc, "imm");
  } else if (p.consumeIfSoftIdentifier("deinit")) {
    handleContextualArgConvention("deinit", kConventionDeinit);
    if (convention == kConventionDeinit && kind != ArgListKind::kArgList) {
      p.emitError(loc,
                  "function types do not support 'deinit'; replace with 'var'")
          << FixIt::replaceToken(loc, "var");
      convention = kConventionVar;
    }
  }

  while (p.getToken().isAny(Token::kw_var, Token::kw_ref)) {
    p.emitTokenError("argument already has a convention specified");
    p.consumeToken();
  }

  markerInfo = KWArgMarkerInfo::kNotMarker;

  // The first token of an argument may be a standalone '*', '/', or '//'
  // marker, and the '*' may also be part of a varargs specification.  Check for
  // these first.
  if (p.consumeIf(Token::slash)) {
    markerInfo = KWArgMarkerInfo::kSlash;
    return success();
  }
  if (p.getToken().isAny(Token::slash_slash)) {
    if (kind != ArgListKind::kParamList &&
        kind != ArgListKind::kFnTypeParamList) {
      p.emitTokenError("'//' is reserved for parameter lists; it marks the end "
                       "of 'infer-only' parameter listings");
    }
    p.consumeToken();
    markerInfo = KWArgMarkerInfo::kSlashSlash;
    return success();
  }
  if (p.consumeIf(Token::star)) {
    if (p.getToken().isAny(Token::comma, Token::r_paren, Token::r_square)) {
      markerInfo = KWArgMarkerInfo::kStar;
      return success();
    }
    variadicKind = VariadicKind::PosVarArg;
  } else if (p.consumeIf(Token::star_star)) {
    variadicKind = VariadicKind::KwVarArg;
    kwArgHandling = KWArgHandling::kKeywordOnly;
  }

  // Reject attempts to make variadic output arguments.
  if (variadicKind != VariadicKind::None && convention == kConventionOut) {
    p.emitError(loc, "'out' convention must not be variadic");
    isErroneous = true;
    variadicKind = VariadicKind::None;
  }

  // Parse the argument name if present.
  if (name) {
    // If we already parsed a name due to lookahead, then we are done.
  } else if (kind == ArgListKind::kFnTypeArgList ||
             kind == ArgListKind::kFnTypeParamList) {
    // When parsing a function type, the name is optional.
    StringAttr maybeArgName;
    if (succeeded(p.parseOptionalIdentifier(maybeArgName, Token::colon)))
      name = maybeArgName;
  } else {
    StringRef argOrParam =
        kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
            ? "parameter"
            : "argument";
    if (p.parseIdentifier(name, "expected " + argOrParam + " name", &loc)) {
      // TODO: Scan ahead for better recovery.
      return failure();
    }
  }

  // Parse an optional type annotation: `":" ["*"] expression`. Omit the colon
  // if a name was not specified.  Bare lambda arg lists do not allow types.
  if (kind != ArgListKind::kBareLambdaArgList) {
    if (p.consumeIf(Token::colon) || !name) {
      SMLoc starLoc = p.getToken().getLoc();
      if (p.consumeIf(Token::star)) {
        if (variadicKind != VariadicKind::PosVarArg) {
          MojoInflightDiag diag = p.emitError(
              starLoc,
              "variadic unpacking with '*' requires a variadic argument");
          if (name) {
            diag.attachNote(loc)
                << "'" << name.getValue() << "' is not a variadic argument";
          }
          return failure();
        }

        if (kind == ArgListKind::kParamList ||
            kind == ArgListKind::kFnTypeParamList) {
          p.emitError(starLoc, "parameters must not be variadic packs");
          return failure();
        }

        variadicKind = VariadicKind::PackVarArg;
      }
      ExprNode *typeExprNode;
      if (p.parseStarredItem(typeExprNode))
        return failure();
      typeExpr = typeExprNode;
    }
  }

  // Set the name to empty string if it wasn't specified.
  if (!name)
    name = StringAttr::get(p.getContext());

  // Parse optional where clauses.
  while (p.getToken().isIdentifier() && p.getToken().getSpelling() == "where") {
    SMLoc whereLoc = p.consumeIdentifier().getLoc();
    if (kind == ArgListKind::kArgList || kind == ArgListKind::kFnTypeArgList) {
      p.emitError(loc,
                  "'where' clauses must be used with parameters and cannot "
                  "be used with arguments");
      return failure();
    }
    if (kind == ArgListKind::kParamList ||
        kind == ArgListKind::kFnTypeParamList) {
      auto diag = p.emitError(
          whereLoc,
          "'where' clauses inside parameter lists are no longer supported");
      if (kind == ArgListKind::kFnTypeParamList) {
        diag.attachNote(whereLoc)
            << "use a trailing 'where' clause after the result type of a "
               "'thin' function type instead";
      } else {
        diag.attachNote(whereLoc)
            << "use a trailing 'where' clause after the signature instead";
      }
    }
    // Parse the constraint for error recovery, then discard it: we already
    // diagnosed it above, and parameter-list `where` clauses are no longer
    // representable. A `where (cond, "msg")` message parses as a single
    // parenthesized expression, so no message handling is needed here.
    ExprNode *discardedProp = nullptr;
    if (p.parseExpression(discardedProp))
      return failure();
  }

  // Parse an optional default argument value: `"=" expression`.
  SMLoc equalLoc;
  if (p.consumeIf(Token::equal, &equalLoc)) {
    if (p.parseExpression(initExpr))
      return failure();

    if (convention == kConventionMut || convention == kConventionOut) {
      p.emitError(equalLoc)
          << (convention == kConventionOut ? "'out'" : "'mut'")
          << " arguments must not have defaults" << initExpr->getRange();
      initExpr = nullptr;
    }

    // Default args and varargs don't mix.
    if (variadicKind != VariadicKind::None) {
      p.emitError(equalLoc, "variadic arguments must not have defaults")
          << initExpr->getRange();
      initExpr = nullptr;
    }
  }

  return success();
}

PassingKind ParsedArgument::getKWArgHandlingAsPassingKind() const {
  // Result slots are not handled through normal call argument resolution.
  if (isResultSlot(kgenConvention))
    return PassingKind::Implicit;

  switch (kwArgHandling) {
  case KWArgHandling::kInferred:
    return PassingKind::Inferred;
  case KWArgHandling::kPositionalOnly:
    return PassingKind::PosOnly;
  case KWArgHandling::kKeywordOnly:
    return PassingKind::KwOnly;
  case KWArgHandling::kPositionalOrKeyword:
    return PassingKind::PosOrKw;
  }
  llvm_unreachable("unhandled KWArgHandling");
}

/// This method handles the function argument list for a Python function.
/// Python has some pretty interesting rules where standalone '*' and '/'
/// markers (when used in place of an argument) actually change the
/// interpretation of other argument definitions by specifying how they behave
/// w.r.t. keyword arguments.  We check these here so the client doesn't
/// have to deal with them.
///
/// This classification logic is described here:
///   https://peps.python.org/pep-0570/#how-to-teach-this
///
/// 'resultArg' is non-null for argument lists, and allows handling of 'out'
/// arguments.
static ParseResult
parseArgOrParamList(ParserBase &p, SmallVectorImpl<ParsedArgument> &parsedArgs,
                    ParsedArgument *resultArg, ArgListKind kind) {
  // Figure out where to stop scanning.
  SmallVector<Token::Kind, 2> stopTokens;
  switch (kind) {
  case ArgListKind::kParamList:
  case ArgListKind::kFnTypeParamList:
    stopTokens.append({Token::r_square, Token::minus_greater});
    break;
  case ArgListKind::kFnTypeArgList:
  case ArgListKind::kArgList:
    stopTokens.push_back(Token::r_paren);
    break;
  case ArgListKind::kBareLambdaArgList:
    stopTokens.push_back(Token::colon);
    break;
  }

  // As we parse all of the arguments and the keyword arguments and markers, we
  // resolve the markers and check the invariants.  Python's parameter grammar
  // embeds checking for `/` and `*` into it, but we do this ad-hoc for
  // simplicity, according to the following rules:
  //
  //   1) Only one '/' and '*' marker may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) `/` cannot be first, and '*' cannot be last in the list.
  //
  // See this for more information:
  // https://peps.python.org/pep-0570/#how-to-teach-this
  bool hasSlashSlashMarker = false, hasSlashMarker = false,
       hasStarMarker = false;
  auto defaultKWArgHandling = KWArgHandling::kPositionalOrKeyword;

  StringRef argOrParam =
      kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
          ? "parameter"
          : "argument";

  // This is invoked when we see a '//' marker.
  auto handleSlashSlashMarker = [&](SMLoc loc) {
    if (hasSlashSlashMarker) {
      p.emitError(loc, "cannot have two '//' markers in the same ")
          << argOrParam << " list";
      return;
    }
    if (hasSlashMarker) {
      p.emitError(loc, "cannot specify '//' marker after '/' marker in ")
          << argOrParam << " list";
      return;
    }
    if (hasStarMarker) {
      p.emitError(loc, "cannot specify '//' marker after '*' marker in ")
          << argOrParam << " list";
      return;
    }
    if (parsedArgs.empty()) {
      p.emitError(loc, "'//' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all parameter we've seen to be inferred only.
    // The remaining ones will stay kPositionalOrKeyword.
    for (ParsedArgument &arg : parsedArgs) {
      arg.kwArgHandling = KWArgHandling::kInferred;
    }

    hasSlashSlashMarker = true;
  };

  // This is invoked when we see a '/' marker.
  auto handleSlashMarker = [&](SMLoc loc) {
    if (hasSlashMarker) {
      p.emitError(loc, "cannot have two '/' markers in the same ")
          << argOrParam << " list";
      return;
    }
    if (hasStarMarker) {
      p.emitError(loc, "cannot specify '/' marker after '*' marker in ")
          << argOrParam << " list";
      return;
    }
    if (parsedArgs.empty()) {
      p.emitError(loc, "'/' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all arguments we've seen that aren't inferred
    // to be positional only. The remaining ones will stay kPositionalOrKeyword.
    for (ParsedArgument &arg : parsedArgs)
      if (arg.kwArgHandling != KWArgHandling::kInferred)
        arg.kwArgHandling = KWArgHandling::kPositionalOnly;
    hasSlashMarker = true;
  };

  // This is invoked when we see a '*' marker or '*arg' argument.
  auto handleStarMarker = [&](SMLoc loc, bool isMarker) -> ParseResult {
    if (hasStarMarker) {
      return p.emitError(loc, "cannot have two '*' markers in the same ")
             << argOrParam << " list";
    }

    // Diagnose '*' marker at end of argument list for completeness.
    if (p.getToken().isAny(stopTokens) && isMarker) {
      p.emitError(loc, "'*' marker is not allowed at end of ")
          << argOrParam << " list";
    }

    // From now on, any parsed arguments are keyword only.
    defaultKWArgHandling = KWArgHandling::kKeywordOnly;
    hasStarMarker = true;

    return success();
  };

  // This parses either an argument or a keyword argument specifier.
  bool foundName = false;
  bool foundKwargs = false;
  auto parseArgument = [&]() -> ParseResult {
    auto marker = KWArgMarkerInfo::kNotMarker;
    ParsedArgument arg;
    arg.kwArgHandling = defaultKWArgHandling;
    if (arg.parse(p, marker, kind))
      return failure();

    // If we have a **arg then it must be the last argument.
    if (foundKwargs) {
      return p.emitError(arg.loc, "'**' marker must be at end of ")
             << argOrParam << " list";
    }

    // If this argument is just a marker, process it.
    if (marker == KWArgMarkerInfo::kSlashSlash) {
      handleSlashSlashMarker(arg.loc);
      return success();
    }
    if (marker == KWArgMarkerInfo::kSlash) {
      handleSlashMarker(arg.loc);
      return success();
    }
    if (marker == KWArgMarkerInfo::kStar)
      return handleStarMarker(arg.loc, /*isMarker=*/true);

    if (arg.name.empty()) {
      if (foundName)
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow named " << argOrParam;

      if (hasSlashMarker || hasStarMarker)
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow '/' or '*'";
    } else {
      foundName = true;
    }

    // Otherwise, if this is a varargs marker, handle it as a marker and an
    // argument.
    if (arg.variadicKind == VariadicKind::PosVarArg ||
        arg.variadicKind == VariadicKind::PackVarArg)
      if (failed(handleStarMarker(arg.loc, /*isMarker=*/false)))
        return failure();

    if (arg.variadicKind == VariadicKind::KwVarArg) {
      foundKwargs = true;

      if (kind == ArgListKind::kParamList ||
          kind == ArgListKind::kFnTypeParamList) {
        return p.emitError(arg.loc,
                           "variadic keyword parameters not supported yet");
      }
      if (arg.convention != ParsedArgument::kConventionUnspec &&
          arg.convention != ParsedArgument::kConventionVar) {
        return p.emitError(
            arg.loc,
            "non-owned variadic keyword arguments are not supported yet");
      }
      if (arg.convention == ParsedArgument::kConventionUnspec) {
        // With no convention written, the argument starts at its '**' token,
        // so the cursor captured at argument start points at it.
        SMLoc starStarLoc = arg.cursor.getToken().getLoc();
        p.emitError(starStarLoc,
                    "variadic keyword arguments only support the 'var' "
                    "convention; add 'var' before '**'")
            << FixIt::insertBeforeToken(starStarLoc, "var ");
        arg.convention = ParsedArgument::kConventionVar;
      }
    }

    // If this argument is an "out" argument, process it as a result.
    if (arg.convention == ParsedArgument::kConventionOut) {
      if (!resultArg)
        return p.emitError(arg.loc, "parameters cannot be 'out'");
      if (resultArg->convention == ParsedArgument::kConventionOut)
        return p.emitError(arg.loc,
                           "function may not have multiple 'out' arguments");
      *resultArg = arg;
      return success();
    }

    // Otherwise just remember the argument.
    parsedArgs.push_back(arg);
    return success();
  };

  // Parse a list of arguments and keyword argument specifiers.  Each argument
  // will leave its `kwargHandling` default initialized.
  if (p.parseCommaSeparatedList(parseArgument, stopTokens))
    return failure();

  // We allow specifying signatures with only positional-only arguments if all
  // the argument names are omitted, i.e. `def(Int, Int) -> Int` is the same as
  // `def(Int, Int, /) -> Int`.
  bool allUnnamedPosOnly = !foundName && !hasSlashMarker && !hasStarMarker;
  for (ParsedArgument &arg : parsedArgs) {
    if (!arg.name.empty() ||
        arg.kwArgHandling == KWArgHandling::kPositionalOnly ||
        arg.kwArgHandling == KWArgHandling::kInferred ||
        arg.variadicKind != VariadicKind::None)
      continue;
    if (!allUnnamedPosOnly)
      return p.emitError(arg.loc, "unnamed ")
             << argOrParam << " must be positional-only";
    arg.kwArgHandling = KWArgHandling::kPositionalOnly;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter signature implementation
//===----------------------------------------------------------------------===//

/// Helper to emit a consistent error message when a required argument or
/// parameter follows a optional one.
static MojoInflightDiag emitOptionalAfterRequired(IREmitter &emitter,
                                                  const ParsedArgument &arg,
                                                  StringRef argOrParam) {
  std::string kindStr = arg.kwArgHandling == KWArgHandling::kKeywordOnly
                            ? "keyword-only"
                            : "positional";
  return emitter.emitError(arg.loc, "required ")
         << kindStr << " " << argOrParam << " follows optional " << kindStr
         << " " << argOrParam << "; change the ordering";
}

/// Helper to emit a default argument/parameter value. Variadic and pack
/// arguments/parameters get a placeholder default iff there are already
/// defaults in the given array of default (i.e. only if a variadic comes after
/// an optional argument/parameter).
/// The default value is returned if available, otherwise a null PValue is
/// returned.
static PValue emitDefault(ArrayRef<ParsedArgument> args, unsigned argIdx,
                          ASTType type, SmallVectorImpl<TypedAttr> &defaults,
                          IREmitter &emitter, ExprContext exprContext) {
  const ParsedArgument &arg = args[argIdx];
  if (const ExprNode *initExpr = arg.initExpr) {
    // If the type is parametric, then we have to be careful about emitting the
    // initializer - we can't check to see if the argument satisfies type
    // conversions until it is used.  However, our IR representation cannot
    // support representing UValues.  If the default value is one, we try to
    // emit it as the parametric argument type anyway.  This resolves more {}'s.
    SmallVector<ParamDeclRefAttr> paramUses;
    emitter.shared.collectParamRefsInType(type, paramUses);

    if (paramUses.empty()) {
      if (auto value = emitter.emitExprPValue(initExpr, exprContext, type))
        return value;
    } else if (auto irValue = emitter.emitExpr(initExpr, exprContext)) {
      // Emit in two stages to handle UValues.
      auto cv = irValue.getIfCValue();
      if (!cv) {
        // If we have a uvalue, we need to emit the UValue as the contextual
        // type to resolve things like {}.
        cv = emitter.emitExprCValue(initExpr, exprContext, type);
      }

      if (cv) {
        if (auto value = emitter.emitPValue({cv, initExpr}, exprContext))
          return value;
      }
    }
    arg.isErroneous = true;
    return UnknownAttr::get(type);
  }

  auto hasAnyPosDefaults = [&]() -> bool {
    return llvm::any_of(llvm::zip(args, defaults), [](auto pair) {
      // Default value on inferred parameters is allowed, since inferred
      // parameters can only be overwritten by keywords, so it does not leads to
      // ambiguity.
      return std::get<0>(pair).kwArgHandling != KWArgHandling::kInferred &&
             std::get<1>(pair) != TypedAttr();
    });
  };

  // If we have a variadic argument, we add a placeholder default value so
  // that invariants about default values always correspond to the trailing
  // arguments. This allows us the have default values before a variadic.
  if (arg.variadicKind != VariadicKind::None && hasAnyPosDefaults())
    return UnknownAttr::get(mlir::NoneType::get(type.mlirType.getContext()));

  // Diagnose an invalid missing default argument: if we have any positional
  // defaults, then we require all the rest to have defaults until the
  // keyword-only section.
  // We allow any keyword-only/inferred parameter to have defaults: They
  // can not cause any ambiguity since they all need to be specified by name.
  if (arg.kwArgHandling != KWArgHandling::kKeywordOnly &&
      arg.kwArgHandling != KWArgHandling::kInferred &&
      arg.kgenConvention != ArgConvention::ByRefResult &&
      arg.kgenConvention != ArgConvention::ByRefError && hasAnyPosDefaults()) {
    MojoInflightDiag diag = emitOptionalAfterRequired(
        emitter, arg,
        exprContext == EC_DefaultParam ? "parameter" : "argument");
    if (arg.typeExpr)
      diag << arg.typeExpr->getRange();
    arg.isErroneous = true;
    return UnknownAttr::get(type);
  }

  // No default value for this argument.
  return PValue();
}

/// Given a type that potentially has all of its parameters unbound, implicitly
/// add the parameter declarations to the function parameters. For example, a
/// struct type can be partially bound. This function implicitly adds a
/// parameter declaration to the function for each unbound struct parameter and
/// binds the struct type to reference those parameters.
///
/// For function types, if the capture origin set parameter is unbound, an
/// implicit parameter for it is added, and a function type of the capture
/// origin set parameter bound to it is returned.
///
/// Parameters can be either added to the end of the parameter as `Implicit`
/// passing-kind parameters if `append` is set (this is used for unbound
/// arguments), or added to the beginning of the parameter list as `Inferred`
/// passing-kind parameters (this is used for unbound parameters).
///
/// If `bindUnboundGeneratorTypes` is set, then we're in a context (like
/// arguments) where we need a concrete type.  This should be false for
/// parameter lists, because they can take unbound (e.g.) function types whose
/// parameters are bound in the body of the declaration.
///
/// On failure, this returns null.
static ASTType addImplicitTypeParams(StringAttr argName, ASTType type,
                                     TypeCheckedParamList &paramList,
                                     bool append, SMLoc loc,
                                     bool bindUnboundGeneratorTypes = true) {
  if (!type) // Propagate an error.
    return {};

  auto &shared = paramList.shared;
  SmallVector<ParamDeclAttr> paramDeclAttrs;
  SmallVector<StringAttr> names;
  SmallVector<PassingKind> passingKinds;
  SmallVector<VariadicKind> variadicKinds;
  SmallVector<SMLoc> locations;
  SmallVector<TypedAttr> defaults;

  // Functor to insert the pending vectors into paramList, either at the front
  // or back.
  auto insertFn = [](size_t insertPt, auto &dst, auto &src) {
    dst.insert(dst.begin() + insertPt, src.begin(), src.end());
  };

  auto commitChanges = llvm::scope_exit([&]() {
    // All lists guaranteed to have the same length.
    if (paramDeclAttrs.empty())
      return;

    // Figure out where to insert the new parameters.  If append, we put them at
    // the end of the list.  If prepend, we put them after any infer-only
    // parameters.
    size_t insertPt;
    if (append)
      insertPt = paramList.paramDeclAttrs.size();
    else {
      insertPt = 0;
      while (insertPt < paramList.paramDeclAttrs.size() &&
             paramList.passingKinds[insertPt] == PassingKind::Inferred)
        ++insertPt;
    }

    insertFn(insertPt, paramList.paramDeclAttrs, paramDeclAttrs);
    insertFn(insertPt, paramList.names, names);
    insertFn(insertPt, paramList.passingKinds, passingKinds);
    insertFn(insertPt, paramList.variadicKinds, variadicKinds);
    insertFn(insertPt, paramList.defaults, defaults);
    insertFn(insertPt, paramList.locations, locations);
  });

  // The parameter decl references that will be used to fully bind the type,
  // plus a parameter evaluator we use to progressively refine the type.
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator = shared.getParameterEvaluator();

  bool hadFailure = false;

  // Re-attach an auto-parameterized generator's trailing `where` (body)
  // constraints to the enclosing parameter list (rebinding their index refs to
  // the just-declared parameters) so they are still checked once the
  // generator's parameters are inferred; must be called after those parameters
  // are declared.
  auto propagateBodyConstraints = [&](PogListAttr srcParamList) {
    for (ConstraintAttr bodyConstraint : srcParamList.getBodyConstraints()) {
      TypedAttr remappedProp =
          evaluator.getReboundAttribute(bodyConstraint.getProposition());
      paramList.emittedBodyConstraints.push_back(ConstraintAttr::get(
          remappedProp, bodyConstraint.getLoc(), bodyConstraint.getMessage()));
    }
  };

  // This functor adds a single parameter to the parameter list.
  auto declareAndAddParam = [&](Type type, StringRef name) {
    auto boundParamType = evaluator.getReboundType(type);
    // If we are prepending this as an implicit parameter, make sure its type
    // doesn't depend on any parameters before it.  Consider something like:
    //   struct T1[mut: Bool, value: TakesBool[mut]]: ...
    //   struct T2[m: Bool, n: T1[m, _]]:
    // it would be incorrect to transform this into:
    //   struct T2[IMP: TakesBool[m], //, m: Bool, n: T1[m, _]]:
    // because IMP would use m before it is declared.
    if (!append) {
      SmallVector<ParamDeclRefAttr> paramUses;
      shared.collectParamRefsInType(boundParamType, paramUses);
      // This is O(n^2) but the N's are small.
      for (ParamDeclRefAttr paramUse : paramUses) {
        // Check to see if it is an earlier part of the type for this argument.
        std::optional<PassingKind> passingKind;
        for (auto [idx, name] : llvm::enumerate(names)) {
          if (paramUse.getName() == name) {
            passingKind = passingKinds[idx];
            break;
          }
        }
        // Otherwise, it must be something part of the enclosing param list.
        if (!passingKind.has_value()) {
          // Make sure to iterate over paramDeclAttrs (not names) because
          // 'names' doesn't get mangled when params shadow.
          for (auto [idx, decl] : llvm::enumerate(paramList.paramDeclAttrs)) {
            if (paramUse.getName() == decl.getName()) {
              passingKind = paramList.passingKinds[idx];
              break;
            }
          }
        }
        // If we didn't find it, check to see if it is on the enclosing struct.
        if (!passingKind.has_value()) {
          auto [curDecl, paramDecls, paramIdx] =
              paramList.declScope.lookupParamReference(paramUse);
          if (curDecl) // Any parameters from it are fine.
            passingKind = PassingKind::Inferred;
        }

        if (!passingKind.has_value()) {
          shared.emitError(loc, "INTERNAL ERROR: inferred parameter of type ")
              << boundParamType << " depends on unresolved parameter "
              << paramUse.getName() << "; please file a compiler bug";
          boundParamType = TypeCheckErrorType::get(shared.getContext());
        } else if (*passingKind != PassingKind::Inferred) {
          // It is common to hit this with variadic packs.
          auto diag = shared.emitError(loc);
          if (auto paramList = dyn_cast<ParamListType>(boundParamType)) {
            diag << "element type parameter " << paramList.getElementType()
                 << " must be an 'inferred' parameter";
          } else {
            diag << "inferred parameter of type " << boundParamType
                 << " cannot depend on non-inferred parameter "
                 << paramUse.getName();
          }
          boundParamType = TypeCheckErrorType::get(shared.getContext());
          hadFailure = true;
        }
      }
    }

    auto mangledName =
        paramList.declScope.mangleParamName(argName.strref() + "." + name);
    auto paramDecl = ParamDeclAttr::get(mangledName, boundParamType);
    names.push_back(mangledName);
    passingKinds.push_back(append ? PassingKind::Implicit
                                  : PassingKind::Inferred);
    paramDeclAttrs.push_back(paramDecl);
    locations.push_back(SMLoc());
    paramValues.push_back(ParamDeclRefAttr::get(paramDecl));

    // FIXME: Autoparam of variadics looks broken?
    variadicKinds.push_back(VariadicKind::None);
    defaults.push_back(TypedAttr());
    evaluator.appendIndexBinding(paramValues.back());
  };

  // First check for a function type.
  // FIXME: We need an AnyFunction metatype.
  if (auto sig = sugarDynCast<FnTypeGeneratorType>(type)) {
    TypedAttr origins = sig.getCaptureOrigins();
    if (!isa<UnboundAttr>(origins))
      return type;
    declareAndAddParam(origins.getType(), "__origins__");
    if (hadFailure)
      return {};
    return sig.getWithCaptureOrigins(paramValues.back());
  }

  if (auto gen = sugarDynCast<GeneratorType>(type)) {
    if (bindUnboundGeneratorTypes) {
      PogListAttr genParamList = gen.getParamListAttrs();
      ArrayRef<PogMetadataAttr> pogs = genParamList.getPogs();
      for (auto [idx, type] : llvm::enumerate(gen.getInputParamTypes()))
        declareAndAddParam(type, pogs[idx].getName());
      if (hadFailure)
        return {};
      propagateBodyConstraints(genParamList);
      return gen
          .getSpecializedGenerator(paramValues, &shared.getEvaluationContext(),
                                   shared.translateLocation(loc))
          .getInstantiatedBody();
    }
  }

  // Auto-parameterize generator typed values.
  if (auto paramType = sugarDynCast<ParamType>(type)) {
    if (auto genType =
            dyn_cast<GeneratorType>(paramType.getParam().getType())) {
      PogListAttr genParamList = genType.getParamListAttrs();
      ArrayRef<PogMetadataAttr> pogs = genParamList.getPogs();
      for (auto [idx, type] : llvm::enumerate(genType.getInputParamTypes()))
        declareAndAddParam(type, pogs[idx].getName());
      if (hadFailure)
        return {};
      propagateBodyConstraints(genParamList);
      TypedAttr generator = paramType.getParam();
      // The generator's trailing constraints were re-attached to the enclosing
      // parameter list above, so they'll be guaranteed to be satisfied under
      // this context. Discharge them all.
      llvm::BitVector dischargedMask(genParamList.getBodyConstraints().size(),
                                     true);
      DenseBoolArrayAttr discharged =
          KGEN::getDenseBoolArrayAttr(generator.getContext(), dischargedMask);
      return BindParamsAttr::get(generator.getContext(), generator, paramValues,
                                 discharged, &shared.getEvaluationContext());
    }
  }

  // Check for a struct type or a struct metatype.
  auto getBoundStructMetaType = [&](StructMetaType metatype) -> StructMetaType {
    // The unbound parameters will be on the struct type's signature.
    TypeSignatureType sig = metatype.getSignature();
    PogListAttr paramList = sig.getParamListAttrs();
    ArrayRef<PogMetadataAttr> pogs = paramList.getPogs();
    for (auto [idx, type] : llvm::enumerate(sig.getParamTypes()))
      declareAndAddParam(type, pogs[idx].getName());
    if (!hadFailure)
      return metatype.bindUnbound(paramValues);
    return {};
  };

  if (auto metatype =
          sugarDynCastIfPresent<StructMetaType>(type.extractMetaType())) {
    auto mt = getBoundStructMetaType(metatype);
    return mt ? mt.getType() : ASTType();
  }
  if (auto metatype = sugarDynCast<StructMetaType>(type))
    return getBoundStructMetaType(metatype);

  return type;
}

// If this argument is a homogenous variadic parameter like "*args: SomeType"
// then the process it into a ParameterList or TypeList.
static ASTType typeCheckVariadicParams(ASTType elementType, ParsedArgument &arg,
                                       IREmitter &emitter,
                                       TypeCheckedParamList &tcParamList) {
  assert(arg.variadicKind == VariadicKind::PosVarArg &&
         "this applies to variadic arguments");

  // Any _'s in the argument type get autoparameterized before we form something
  // around it.
  elementType = addImplicitTypeParams(arg.name, elementType, tcParamList,
                                      /*append=*/true, arg.loc);
  if (!elementType)
    return {};

  // Form a ParameterList/TypeList type.
  ASTType listType;
  auto elementMetaType = elementType.extractMetaType();

  bool isValueList = sugarIsa<StructMetaType>(elementMetaType) ||
                     sugarIsa<NonStructTypeType>(elementMetaType) ||
                     sugarIsa<TraitType>(elementMetaType);
  if (isValueList)
    listType = emitter.shared.lookupBuiltinType("ParameterList",
                                                emitter.declScope, arg.loc);
  else
    listType = emitter.shared.lookupBuiltinType("TypeList", emitter.declScope,
                                                arg.loc);

  if (isa<TypeCheckErrorType>(listType))
    return listType; // Sanity check the returned VariadicList declaration.
  ASTDecl *listDecl = listType.getDecl(emitter.shared);

  // We expect:
  //   ParameterList[type: AnyType, //, *values: type](
  if (!listDecl) {
    emitter.emitError(arg.loc, "malformed ParameterList");
    return {};
  }
  auto structDeclOp =
      dyn_cast_if_present<StructDeclOp>(*listDecl->getIfOperation());
  if (!structDeclOp || structDeclOp.getParams().size() != 2) {
    emitter.emitError(arg.loc, "malformed ParameterList");
    return {};
  }

  ParamBindings bindings(emitter.declScope, arg.typeExpr);
  bindings.add(
      arg.typeExpr, PValue(elementType),
      StringAttr::get(emitter.getContext(), isValueList ? "type" : "Trait"));
  bindings.add(arg.typeExpr, // 'values' is left unbound.
               UnboundAttr::get(UnresolvedType::get(emitter.getContext())),
               StringAttr::get(emitter.getContext(), "values"));

  TypeSignatureType sig = structDeclOp.getSignature();
  ParamInf inference(bindings, sig.getParamTypes(), sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, listDecl,
                     /*discardError=*/false);
  VerifiedParamBindings verifiedBindings = inference.inferForStruct();

  if (!verifiedBindings)
    return emitter.shared.getTypeCheckErrorType();
  ASTType result = verifiedBindings.specializeStructType(structDeclOp);

  // Add the !kgen.param_list parameter to the parameter list.  It is possible
  // the element type is a non-inferred parameter, so "append" this.
  return addImplicitTypeParams(arg.name, result, tcParamList,
                               /*append=*/false, arg.loc);
}

TypeCheckedParamList::TypeCheckedParamList(ASTDecl &declScope)
    : declScope(declScope), shared(declScope.getShared()) {}

std::optional<TypeCheckedParamList>
TypeCheckedParamList::create(ParsedParamList &parsedParams,
                             ASTDecl &declScope) {
  TypeCheckedParamList result(declScope);
  result.stagedBodyConstraints = std::move(parsedParams.bodyConstraints);

  // Resolve each of the parameter declarations.
  IREmitter emitter(declScope, EC_Type, &result.deferredTypingContext);
  bool hasErrors = false;

  IndexRefRemapper remapper(ArrayRef<ParamDeclAttr>{});
  for (auto [argIdx, arg] : llvm::enumerate(parsedParams.params)) {
    // Check for things supported in arguments that are not supported in
    // parameters.
    ASTType type;
    if (arg.typeExpr) {
      type = emitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);

      auto fnType = dyn_cast<FnTypeGeneratorType>(type);
      auto *fnTypeExpr = dyn_cast<FunctionTypeNode>(arg.typeExpr);
      if (fnType && fnTypeExpr && !fnTypeExpr->isThin &&
          !fnTypeExpr->effects.isCapturing()) {
        ASTDecl *closureTrait = result.shared.getOrCreateClosureTrait(
            declScope.getLoc(), *declScope.getNearestDeclOfType<FileModuleOp>(),
            fnType);
        type = TraitType::get(getFullyResolvedSymbolRef(
            cast<mlir::SymbolOpInterface>(closureTrait->getIfOperation())));
      }
    } else {
      emitter.emitError(arg.loc, "parameters must always have a type");
      arg.isErroneous = true;
      hasErrors = true;
    }
    if (!type) {
      type = emitter.shared.getTypeCheckErrorType();
      arg.variadicKind = VariadicKind::None;
      hasErrors = true;
    }

    // We type check variadicKind and may need to change it, so copy it.
    auto variadicKind = arg.variadicKind;
    assert(variadicKind != VariadicKind::PackVarArg &&
           "parameters may not be variadic packs");

    // Variadics turn into ParameterList or TypeList.
    if (variadicKind == VariadicKind::PosVarArg) {
      type = typeCheckVariadicParams(type, arg, emitter, result);
      if (!type)
        arg.variadicKind = VariadicKind::None;
    }

    type = addImplicitTypeParams(arg.name, type, result,
                                 /*append=*/false, arg.loc,
                                 /*bindUnboundGeneratorTypes=*/false);
    if (!type) {
      type = emitter.shared.getTypeCheckErrorType();
      hasErrors = true;
    }

    // Emit default parameter values if present. An error would have been
    // emitted if failed.
    PValue defaultVal = emitDefault(parsedParams.params, argIdx, type,
                                    result.defaults, emitter, EC_DefaultParam);
    result.defaults.push_back(defaultVal);

    // TODO: Parameter decls should support conventions at some point.
    if (arg.convention != ParsedArgument::kConventionUnspec) {
      emitter.emitError(arg.loc, "parameters must always be passed by-value");
      hasErrors = true;
    }

    // Bind the parsed type expression so references from other parameters
    // can be resolved. The parameter names in ParamDeclAttr are mangled with
    // the location so that parameter names in mojo are unique in the IR.
    auto newDecl = ParamDeclAttr::get(
        declScope.mangleUserDefinedParamName(arg.name), type);
    result.paramDeclAttrs.push_back(newDecl);
    result.locations.push_back(arg.loc);

    // The unmangled names are also collected to aid keyword parameter binding.
    result.passingKinds.push_back(arg.getKWArgHandlingAsPassingKind());
    result.names.push_back(arg.name);
    result.variadicKinds.push_back(variadicKind);

    ASTDecl &resolvedDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        PValue(ParamDeclRefAttr::get(newDecl)), arg.name, arg.loc, &declScope);
    emitter.shared.notifyListenerOnParameterDecl(resolvedDecl, arg.loc);
    remapper.appendParamDecl(newDecl);
  }

  if (hasErrors)
    return std::nullopt;
  return result;
}

void TypeCheckedParamList::emitBodyConstraints() {
  SharedState &shared = declScope.getShared();
  IREmitter constraintEmitter(declScope, EC_Requires);
  for (const ParsedConstraint &constraint : stagedBodyConstraints) {
    RValue prop =
        constraintEmitter.emitExprScalarBool(constraint.propExpr, EC_Requires);
    if (!prop) {
      constraintEmitter.emitError(constraint.loc,
                                  "failed to emit constraint expression");
      continue;
    }

    PValue propVal = prop.getIfPValue();
    if (!propVal) {
      constraintEmitter.emitErrorForDynamicValueInParameter(constraint.loc);
      continue;
    }

    // Convert `x and y` to `x & y` so we get better canonicalization.
    propVal = deShortCircuitCond(propVal);

    // Translate location without any DebugInfo scope since this metadata is
    // purely frontend use and never ends up in DWARF.
    auto bodyConstraint = ConstraintAttr::get(
        propVal, shared.diags.translateLocation(constraint.loc),
        constraint.message);
    emittedBodyConstraints.push_back(bodyConstraint);

    // Insert the constraint into the param-list's declScope immediately
    // so that subsequent constraint expressions can reference it.
    declScope.insertKnownAssumptions({bodyConstraint});
  }

  // Now that all body constraints have been emitted and added as known
  // assumptions on `declScope`, try to discharge each body constraint that
  // was deferred while emitting parameter declaration types and the function
  // signature. A deferred constraint is "discharged" if the body constraints
  // (together with the rest of the scope's known assumptions) now imply it;
  // otherwise we surface a hard error pointing at the original binding site.
  for (const DeferredConstraint &deferral :
       deferredTypingContext.deferredConstraints) {
    SmallVector<ConstraintAttr> stillUnprovable;
    std::optional<MojoInflightDiag> violationDiag;
    auto getDiag = [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
      violationDiag = shared.emitError(loc ? *loc : deferral.deferralLoc);
      return *violationDiag;
    };
    TriBool result = LIT::canDischargeConstraintsInScope(
        declScope, /*paramListAttr=*/PogListAttr(), {deferral.constraint},
        /*origConstraints=*/{}, getDiag, &stillUnprovable,
        /*evaluator=*/nullptr);
    if (result.isTrue())
      continue;

    if (result.isUnknown()) {
      MojoInflightDiag diag = shared.emitError(deferral.deferralLoc)
                              << "invalid bindings in signature: lacking "
                                 "evidence to prove correctness";
      for (ConstraintAttr unprovable : stillUnprovable) {
        LIT::emitConstraintInconclusive(shared.getDeclResolver(), diag,
                                        unprovable);
        // Point the user at the signature they're defining.
        diag.attachNote(declScope.getLoc())
            << "add a trailing 'where' clause that requires "
            << unprovable.getProposition();
      }
    }
    // For `Violated`, `getDiag` was invoked and
    // `canDischargeConstraintsInScope` populated `violationDiag` with a
    // "violated constraint" message plus per-constraint notes. It will commit
    // when `violationDiag` goes out of scope.
  }
}

PogListAttr TypeCheckedParamList::getParamListAttr() const {
  // In a parameter list, any variadic list is claimed to be ImmMem.
  std::optional<ArgConvention> origVariadicConvention;
  for (auto var : variadicKinds) {
    if (var != VariadicKind::None)
      origVariadicConvention = ArgConvention::ImmMem;
  }

  return PogListAttr::get(shared.getContext(), names, passingKinds,
                          variadicKinds, defaults, origVariadicConvention,
                          emittedBodyConstraints);
}

//===----------------------------------------------------------------------===//
// ParsedConstraint Implementation
//===----------------------------------------------------------------------===//

ParseResult ParsedConstraint::parse(ParserBase &p) {
  loc = p.getToken().getLoc();

  // Parse the constraint expression into a local; `extractParenthesizedMessage`
  // splits it into the final `propExpr` (the condition) and `message`.
  ExprNode *parsed;
  if (p.parseExpression(parsed))
    return failure();

  // A message is written `where (condition, "message")`, which the expression
  // parser produces as a parenthesized two-element tuple; split it if present.
  return extractParenthesizedMessage(p, parsed);
}

ParseResult ParsedConstraint::extractParenthesizedMessage(ParserBase &p,
                                                          ExprNode *parsed) {
  // A message clause has the shape `where (condition, "message")`. The
  // expression parser produces a ParenNode wrapping a two-element TupleNode
  // for this. Anything else (a bare condition, or a parenthesized condition
  // like `where (a and b)`) is the condition itself.
  auto *paren = dyn_cast<ParenNode>(parsed);
  if (!paren) {
    propExpr = parsed;
    return success();
  }
  auto *tuple = dyn_cast<TupleNode>(paren->subExpr);
  if (!tuple) {
    propExpr = parsed;
    return success();
  }

  // A tuple with the wrong arity can only be a mistyped message clause: a
  // condition is a scalar bool, never a tuple, so `where (a, b, c)` is not a
  // valid condition either. Give a targeted diagnostic instead of letting it
  // fall through to a generic "not scalar<bool>" error.
  if (tuple->exprs.size() != 2)
    return p.emitError(tuple->getLoc(),
                       "a 'where' clause takes a condition and an optional "
                       "message: 'where (condition, \"message\")'");

  // The second element is the message. For now only string literals are
  // supported: a `where` message must be available in the parser, but a
  // non-literal expression would need comptime evaluation that the parser
  // cannot perform. Reject non-literal messages with a targeted diagnostic.
  ExprNode *msgExpr = tuple->exprs[1];
  auto *strLit = dyn_cast<StringLiteralNode>(msgExpr);
  if (!strLit)
    return p.emitError(msgExpr->getLoc(),
                       "the message in a 'where' clause must be a string "
                       "literal");

  // `getValue()` already handles adjacent string-literal concatenation.
  message = StringAttr::get(p.getContext(), strLit->getValue());
  propExpr = tuple->exprs[0];
  return success();
}

//===----------------------------------------------------------------------===//
// ParsedParamList Implementation
//===----------------------------------------------------------------------===//

/// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
/// param_list   ::= argument_list | "(" ")"
/// param_result_types ::= expression ("," expression)*
ParseResult ParsedParamList::parseParametersIfPresent(ParserBase &p,
                                                      ArgListKind kind) {
  // Check to see if a parameter signature exists at all.
  if (!p.consumeIf(Token::l_square) || p.consumeIf(Token::r_square))
    return success();

  // Parse an actual parameter list.
  if (parseArgOrParamList(p, params, /*resultArg=*/nullptr, kind))
    return failure();

  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

ParseResult ParsedParamList::parseTrailingConstraintsIfPresent(ParserBase &p) {
  while (p.consumeIfSoftIdentifier("where")) {
    ParsedConstraint constraint;
    if (constraint.parse(p))
      return failure();

    bodyConstraints.push_back(constraint);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Capture Signature Parsing
//===----------------------------------------------------------------------===//

// capture_list ::= "{" capture_item ("," capture_item)* [","] "}" | "{}"
// capture_item ::= capture_convention [identifier^] | identifier | identifier^
ParseResult ParsedCaptureList::parseCaptureList(ParserBase &p) {
  if (!p.consumeIf(Token::l_brace))
    return success();
  hasExplicitCaptureList = true;
  auto parseConvention = [&]() -> CaptureConvention {
    SMLoc convLoc = p.getToken().getLoc();
    if (p.consumeIf(Token::kw_var)) {
      if (p.consumeIf(Token::caret))
        return CaptureConvention::kConventionMove;
      else
        return CaptureConvention::kConventionCopy;
    } else if (p.consumeIfSoftIdentifier("mut")) {
      return CaptureConvention::kConventionMut;
    } else if (p.consumeIfSoftIdentifier("imm")) {
      return CaptureConvention::kConventionRead;
    } else if (p.consumeIfSoftIdentifier("read")) {
      p.emitError(convLoc, "'read' was removed; use 'imm'")
          << FixIt::replaceToken(convLoc, "imm");
      return CaptureConvention::kConventionRead;
    } else if (p.consumeIf(Token::kw_ref)) {
      return CaptureConvention::kConventionRef;
    }

    return CaptureConvention::kConventionUnspecified;
  };

  auto isDuplicateCapture = [&](StringRef name) {
    return llvm::any_of(parsedCaptures, [&](const auto &existing) {
      return std::get<0>(existing) == name;
    });
  };

  auto parseArgument = [&]() -> ParseResult {
    SMLoc captureLocation = p.getToken().getLoc();
    CaptureConvention parsedConvention = parseConvention();
    StringAttr nameValue;
    if (succeeded(p.parseOptionalIdentifier(nameValue))) {
      if (isDuplicateCapture(nameValue.getValue())) {
        return p.emitError(captureLocation, "duplicate capture of '")
               << nameValue.getValue() << "'; remove the duplicate entry";
      }
      if (p.getToken().is(Token::comma) || p.getToken().is(Token::r_brace)) {
        CaptureConvention convention = [parsedConvention]() {
          if (parsedConvention == CaptureConvention::kConventionUnspecified)
            return CaptureConvention::kConventionRead;
          return parsedConvention;
        }();

        parsedCaptures.push_back(
            {nameValue.getValue(), convention, captureLocation});
        return success();
      }
      // consume '^'
      if (p.consumeIf(Token::caret)) {
        // This has to be either `{var x^}` or a `{x^}`
        if (parsedConvention != CaptureConvention::kConventionCopy &&
            parsedConvention != CaptureConvention::kConventionUnspecified) {
          p.emitError(captureLocation,
                      "'^' requires 'var' convention; write 'var x^' to move "
                      "a capture");
          return failure();
        }
        // Refines the convention to move.
        parsedCaptures.push_back({nameValue.getValue(),
                                  CaptureConvention::kConventionMove,
                                  captureLocation});
        return success();
      }
      return p.emitError(captureLocation,
                         "capture lists expect references to variables");
    }

    // We parsed a convention but there is no identifier after it, then it must
    // be a default capture.
    if (parsedConvention != CaptureConvention::kConventionUnspecified) {
      if (captureAllByConvention.has_value()) {
        auto diag = p.emitError(captureLocation,
                                "default capture convention was already "
                                "specified; remove the duplicate");
        diag.attachNote(captureLocation)
            << "a capture convention (like 'mut' or 'var') before the "
               "capture list sets the default for all captured variables";
        return failure();
      }

      captureAllByConvention = parsedConvention;
      return success();
    }

    return p.emitError(captureLocation, "expected a capture convention "
                                        "(like 'mut' or 'var')");
  };

  if (!p.consumeIf(Token::r_brace)) {
    do {
      // Enable trailing commas
      if (p.consumeIf(Token::r_brace))
        return success();
      if (parseArgument())
        return failure();
    } while (p.consumeIf(Token::comma));
    if (p.parseToken(Token::r_brace, "expected '}' in capture list"))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Function Signature Parsing
//===----------------------------------------------------------------------===//

/// Parse an argument list, including the parentheses around them.  This also
/// parses 'raises' and other effects.
ParseResult ParsedArgumentList::parseArgumentListAndEffects(ParserBase &p,
                                                            ArgListKind kind) {

  // If this is a bare lambda argument list, it won't be parenthesized and won't
  // have effects.
  if (kind == ArgListKind::kBareLambdaArgList)
    return parseArgOrParamList(p, parsedArgs, &resultArg, kind);

  if (p.parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();

  if (!p.consumeIf(Token::r_paren)) {
    if (parseArgOrParamList(p, parsedArgs, &resultArg, kind) ||
        p.parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  auto isEffectKeywordOrWhere = [&](StringRef spelling) {
    return spelling == "raises" || spelling == "capturing" ||
           spelling == "escaping" || spelling == "thin" ||
           spelling == "register_passable" || spelling == "abi" ||
           spelling == "where" ||
           // TODO: remove this after the parametric closure trait become
           // default.
           spelling == "__param_trait__";
  };

  // If the client supports function effects, parse them as well.
  // Parse other function effects.
  while (p.getToken().isIdentifier()) {
    SMLoc loc = p.getToken().getLoc();
    StringRef spelling = p.getTokenSpelling();

    auto handleEffect = [&](auto hasFn, auto setFn) {
      if ((effects.*hasFn)())
        p.emitError(loc, "function effect '")
            << spelling << "' was already specified; remove the duplicate";
      (effects.*setFn)(true);
    };

    if (!isEffectKeywordOrWhere(spelling)) {
      // If this isn't a known effect, then it could be an error like a missing
      // colon at the end of a function declaration.  If so, emit a nice error
      // and recover cleanly.
      if (p.getToken().isStartOfLine() && kind == ArgListKind::kArgList) {
        // Otherwise maybe it was misspelled, just eat it.
        p.emitError(p.getTokenLocOrEndOfPreviousLineIfOnNewLine(),
                    "missing ':' at end of function signature");
        return failure();
      }

      // Otherwise maybe it was misspelled, just eat it.
      p.emitError(loc, "unknown function effect '")
          << spelling << "', expected 'raises', 'capturing', or 'thin'";
    } else if (spelling == "raises") {
      handleEffect(&FnEffects::isThrows, &FnEffects::setThrows);
      p.consumeIdentifier();

      // A "C" ABI function cannot propagate a Mojo error across the C boundary.
      if (effects.isCABI()) {
        p.emitError(loc) << "'abi(\"C\")' function may not be marked 'raises'; "
                            "remove 'raises' or use 'abi(\"Mojo\")'";
        return failure();
      }

      // Parse a thrown type if specified.  Signatures can exist in function
      // declarations but also in function types like "def () raises X", so
      // we have to be careful about parsing the thing after 'raises' as a
      // thrown type when it isn't.  The thrown type is any primary
      // expression (a dotted name, `Self.AssocType`, a parenthesized union,
      // etc.), so the guard should mirror the set of tokens that
      // `parsePrimaryExpr` itself accepts.  Two exceptions:
      //   - Another effect keyword (`capturing`, `where`, ...) means there
      //     is no thrown type — the outer loop handles the next effect.
      //   - `{` starts a `raises {captures}` capture list, parsed later by
      //     `parseCaptureList`; it must not be consumed here as a set
      //     literal thrown type.
      Token::Kind nextKind = p.getToken().getKind();
      if (!isEffectKeywordOrWhere(p.getTokenSpelling()) &&
          !p.getToken().isStartOfLine() && nextKind != Token::l_brace &&
          ParserBase::isPrimaryExprStart(nextKind)) {
        (void)p.parseExpression(thrownTypeExpr, /*stmtIndent=*/std::nullopt,
                                ParserBase::Precedence::kPrimary);
      }
      continue; // Don't consume the identifier again.

    } else if (spelling == "capturing") {
      handleEffect(&FnEffects::isCapturing, &FnEffects::setCapturing);
    } else if (spelling == "escaping") {
      p.emitError(loc, "the 'escaping' function effect is no longer supported");
      return failure();
    } else if (spelling == "thin") {
      if (kind != ArgListKind::kFnTypeArgList) {
        p.emitError(loc, "function effect 'thin' must only be used on function "
                         "types");
      } else if (isThin) {
        p.emitError(loc, "function effect 'thin' was already specified; remove "
                         "the duplicate");
      }
      isThin = true;
    } else if (spelling == "__param_trait__") {
      assert((kind == ArgListKind::kFnTypeArgList ||
              kind == ArgListKind::kArgList) &&
             "__param_trait__ must only be used on function types");
      isExperimentalParamTrait = true;
    } else if (spelling == "register_passable") {
      p.emitWarning(loc)
          << "the 'register_passable' function effect is no longer supported; "
             "use trait constraints like "
             "'RegisterPassable & def(...) -> ...' instead";
      p.consumeIdentifier();
      continue;
    } else if (spelling == "abi") {
      p.consumeIdentifier(); // consume 'abi'
      if (p.parseToken(Token::l_paren, "expected '(' after 'abi'"))
        return failure();
      if (!p.getToken().is(Token::string)) {
        p.emitError(p.getToken().getLoc(),
                    "expected calling convention string after 'abi('");
        return failure();
      }
      StringRef conv = p.getTokenSpelling(); // includes quotes: "\"C\""
      // Strip the surrounding quotes to get the bare convention name.
      StringRef convName =
          conv.size() >= 2 ? conv.drop_front().drop_back() : conv;
      if (!convName.equals_insensitive("C") &&
          !convName.equals_insensitive("Mojo")) {
        p.emitError(p.getToken().getLoc(), "unsupported calling convention ")
            << conv << ", expected \"C\" or \"Mojo\"";
        return failure();
      }
      bool isCABI = convName.equals_insensitive("C");
      p.consumeToken(); // consume the string literal
      if (p.parseToken(Token::r_paren,
                       isCABI ? "expected ')' after 'abi(\"C\"'"
                              : "expected ')' after 'abi(\"Mojo\"'"))
        return failure();
      if (hasExplicitABI)
        p.emitError(
            loc, "'abi()' effect was already specified; remove the duplicate");
      hasExplicitABI = true;
      if (isCABI) {
        effects.setCABI(true);

        // A "C" ABI function cannot propagate a Mojo error across the C
        // boundary.
        if (effects.isThrows()) {
          p.emitError(loc)
              << "'abi(\"C\")' function may not be marked 'raises'; "
                 "remove 'raises' or use 'abi(\"Mojo\")'";
          return failure();
        }
      }
      // abi("Mojo") is the default Mojo calling convention — recorded as
      // explicit but leaves the CABI bit unset.
      continue; // tokens already consumed; skip bottom p.consumeIdentifier()
    } else {
      assert(spelling == "where" && "isEffectKeywordOrWhere unknown keyword");
      break;
    }

    p.consumeIdentifier();
  }

  return success();
}

/// Parse the result specifier starting with a `->` if present.
void ParsedArgumentList::parseResultIfPresent(
    ParserBase &p, std::optional<size_t> stmtIndent) {
  SMLoc arrowLoc;
  if (!p.consumeIf(Token::minus_greater, &arrowLoc)) {
    // Make sure the result arg has a location of the end of the argument if not
    // specified by an 'out' argument, so that synthesized results (none etc)
    // have a location.
    if (!resultArg.loc.isValid())
      resultArg.loc = p.getToken().getLoc();
    return;
  }

  // We may have already parsed an 'out' argument.  If so, this will be an error
  // and we may want to undo things.
  auto oldResultArg = resultArg;
  resultArg.loc = p.getToken().getLoc();

  // Parse a result reference if present.
  bool isRefResult = false;
  if (p.consumeIf(Token::kw_ref)) {
    if (succeeded(p.parseRefSpecifier(resultArg.refOriginExpr))) {
      if (resultArg.refOriginExpr) {
        isRefResult = true;
      } else {
        p.emitError(resultArg.loc, "'ref' result requires an origin specifier");
      }
    }
  }

  // Parse the result type expression.
  // If this result parsing fails, then we just continue on as if none was
  // specified.
  (void)p.parseExpression(resultArg.typeExpr, stmtIndent);

  // If we already had a result, emit an error but keep parsing.
  if (resultArg.convention == ParsedArgument::kConventionOut) {
    auto diag = p.emitError(resultArg.loc)
                << "functions must not declare both an 'out' argument "
                   "and a return type";
    // It is common to include -> None on initializers, provide a helpful
    // message.
    if (resultArg.typeExpr &&
        resultArg.typeExpr->kind == ExprNode::kNoneLiteral) {
      diag << "; remove the '-> None' to fix it";
      diag.addFixIt(FixIt::remove(
          SourceRange(arrowLoc, resultArg.typeExpr->getRangeEnd())));
      resultArg = oldResultArg;
    }
  }

  // Indicate a present result by setting its convention to 'out' or 'ref'.
  resultArg.convention = isRefResult ? ParsedArgument::kConventionRef
                                     : ParsedArgument::kConventionOut;
}

/// This function creates a new anonymous origin decl for the specified
/// argument, and wraps the type with a RefType using that origin.
static RefType makeImplicitRefTypeForArg(const ParsedArgument &arg, size_t idx,
                                         Type type, bool isMutable,
                                         TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;

  StringAttr originName;
  if (arg.name) {
    originName = declScope.mangleParamName(arg.name.strref());
  } else { // Used by function types, for example.
    originName =
        declScope.mangleParamName(Twine(llvm::utostr(idx)) + "_unnamed");
  }

  auto originDecl = ParamDeclAttr::get(
      originName, OriginType::get(originName.getContext(), isMutable));

  // Tell the signature about the new origin decl.
  tcSignature.implicitOriginDecls.push_back(originDecl);

  return RefType::get(type,
                      ParamDeclRefAttr::get(originName, originDecl.getType()));
}

// If this argument is a pack vararg like "*args: *Ts" then the argument
// expression is "Ts", and the star before it was syntactically parsed.
// This expression must be a PValue of variadic metatype.  We need to
// process it into a VariadicPack.
static ASTType typeCheckVariadicPack(ParsedArgument &arg, size_t argIdx,
                                     IREmitter &emitter,
                                     TypeCheckedFnSignature &tcSignature) {
  assert(arg.variadicKind == VariadicKind::PackVarArg &&
         "this applies to pack arguments");

  PValue param = emitter.emitExprPValue(arg.typeExpr, EC_Type);
  if (!param) // Error emitting the expression is already diagnosed.
    return {};

  /// Check for autoparameterization of the type list.
  auto paramType = param.getRValueType();
  paramType = addImplicitTypeParams(arg.name, paramType, tcSignature.paramList,
                                    /*append=*/true, arg.loc);
  if (!paramType.isEqualCanon(param.getRValueType())) {
    // If the type list is autoparameterized, rebuild the value so it binds with
    // the new parameter type correctly.
    param = PValue(LIT::getEmptyStructValue(paramType));
  }

  // Make sure the param value is a variadic list of types.
  auto elementType = paramType.getParameterListInfo().elementType;
  if (!elementType) {
    emitter.emitError(arg.typeExpr->getLoc(),
                      "pack argument type list must reference a variadic list")
        << arg.typeExpr->getRange();
    return {};
  }

  // Form a VariadicPack type.  Note that we cannot use ParamBindings to do this
  // as we have no way to "splat" the type list into the variadic list :-(.
  ASTType variadicPackType = emitter.shared.lookupBuiltinType(
      "VariadicPack", emitter.declScope, arg.loc);
  if (isa<TypeCheckErrorType>(variadicPackType))
    return {}; // Sanity check the returned VariadicPack declaration.
  ASTDecl *packDecl = variadicPackType.getDecl(emitter.shared);

  // We expect:
  // VariadicPack[
  //   elt_is_mutable: Bool, _mlir_origin: !lit.origin, origin: Origin[mut],
  //   element_trait: type_of(AnyType), //, is_owned: Bool, *element_types:
  //   element_type]
  if (!packDecl) {
    emitter.emitError(arg.loc, "malformed VariadicPack");
    return {};
  }
  auto packStruct =
      dyn_cast_if_present<StructDeclOp>(*packDecl->getIfOperation());
  if (!packStruct || packStruct.getParams().size() != 7) {
    emitter.emitError(arg.loc, "malformed VariadicPack");
    return {};
  }

  ParamBindings bindings(emitter.declScope, arg.typeExpr);
  // The reference is immutable when borrowing, mutable otherwise.
  bool isMutable = arg.convention != ParsedArgument::kConventionImm &&
                   arg.convention != ParsedArgument::kConventionUnspec;
  bindings.add(arg.typeExpr,
               SIMDAttr::getScalarBool(emitter.getContext(), isMutable),
               StringAttr::get(emitter.getContext(), "elt_is_mutable"));
  bindings.add(arg.typeExpr,
               UnboundAttr::get(UnresolvedType::get(emitter.getContext())),
               StringAttr::get(emitter.getContext(), "origin"));
  bindings.add(arg.typeExpr, PValue(elementType),
               StringAttr::get(emitter.getContext(), "element_trait"));

  bool isOwned = arg.convention == ParsedArgument::kConventionVar;
  bindings.add(arg.typeExpr,
               SIMDAttr::getScalarBool(emitter.getContext(), isOwned));

  // Splat in the list of types.
  bindings.add(arg.typeExpr,
               UnpackedAttr::get(param, /*kwOnly=*/false, elementType));

  TypeSignatureType sig = packStruct.getSignature();
  ParamInf inference(bindings, sig.getParamTypes(), sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, packDecl,
                     /*discardError=*/false);
  VerifiedParamBindings verifiedBindings = inference.inferForStruct();
  if (!verifiedBindings)
    return {};
  return verifiedBindings.specializeStructType(packStruct);
}

// If this argument is a homogenous vararg like "*args: SomeType" then the
// process it into a VariadicList.
static ASTType typeCheckVariadicList(ParsedArgument &arg, IREmitter &emitter,
                                     TypeCheckedFnSignature &tcSignature) {
  assert(arg.variadicKind == VariadicKind::PosVarArg &&
         "this applies to variadic list arguments");

  ASTType elementType =
      emitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);
  if (!elementType) // Error emitting the expression is already diagnosed.
    return {};

  // Any _'s in the argument type get autoparameterized before we form the
  // VariadicList around it.
  elementType =
      addImplicitTypeParams(arg.name, elementType, tcSignature.paramList,
                            /*append=*/true, arg.loc);

  // Form a VariadicList type.
  ASTType variadicListType = emitter.shared.lookupBuiltinType(
      "VariadicList", emitter.declScope, arg.loc);
  if (isa<TypeCheckErrorType>(variadicListType))
    return {}; // Sanity check the returned VariadicList declaration.
  ASTDecl *listDecl = variadicListType.getDecl(emitter.shared);

  // We expect:
  // VariadicList[elt_is_mutable: Bool, mlir_origin, origin: Origin,
  //              element_type: AnyType, is_owned: Bool]
  if (!listDecl) {
    emitter.emitError(arg.loc, "malformed VariadicList");
    return {};
  }
  auto structDeclOp =
      dyn_cast_if_present<StructDeclOp>(*listDecl->getIfOperation());
  if (!structDeclOp || structDeclOp.getParams().size() != 5) {
    emitter.emitError(arg.loc, "malformed VariadicList");
    return {};
  }

  ParamBindings bindings(emitter.declScope, arg.typeExpr);
  // The reference is immutable when borrowing, mutable otherwise.
  bool isMutable = arg.convention != ParsedArgument::kConventionImm &&
                   arg.convention != ParsedArgument::kConventionUnspec;
  // See typeCheckVariadicPack above for why we use scalar<bool> instead of i1.
  bindings.add(arg.typeExpr,
               SIMDAttr::getScalarBool(emitter.getContext(), isMutable),
               StringAttr::get(emitter.getContext(), "elt_is_mutable"));

  bindings.add(arg.typeExpr, // Origin is left unbound.
               UnboundAttr::get(UnresolvedType::get(emitter.getContext())),
               StringAttr::get(emitter.getContext(), "origin"));
  bindings.add(arg.typeExpr, PValue(elementType));

  bool isVar = arg.convention == ParsedArgument::kConventionVar;
  bindings.add(arg.typeExpr,
               SIMDAttr::getScalarBool(emitter.getContext(), isVar));

  TypeSignatureType sig = structDeclOp.getSignature();
  ParamInf inference(bindings, sig.getParamTypes(), sig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, listDecl,
                     /*discardError=*/false);
  VerifiedParamBindings verifiedBindings = inference.inferForStruct();

  if (!verifiedBindings)
    return {};
  return verifiedBindings.specializeStructType(structDeclOp);
}

/// Type check each argument in turn, resolving their type and default
/// initializer value.  Arguments in Mojo can refer to previous arguments in
/// their type+default value expressions as PValues, so we need to ensure that
/// they are emitted and have declarations registered in the scope so that later
/// lookups can find them.
static void typeCheckOneArgument(size_t idx, ASTDecl *fnDecl,
                                 TypeCheckedFnSignature &tcSignature) {
  ParsedArgument &arg = tcSignature.argList.parsedArgs[idx];

  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = declScope.getShared();
  IREmitter typeEmitter(declScope, EC_Type,
                        &tcSignature.paramList.deferredTypingContext);

  FnOp fnOp; // Null if type checking a function type.
  if (fnDecl)
    fnOp = cast<FnOp>(*fnDecl->getIfOperation());

  // True if this is a static method.
  // FIXME: This is completely wrong, @static_method decorator hasn't been
  // applied yet.
  //
  // It isn't clear if this is actually that bad, maybe we should just say that
  // first arguments in methods default to Self it they don't have type.  This
  // could be true for static methods as well.
  bool isStaticMethod = tcSignature.selfType && fnOp.getIsStatic();

  // Start by computing the declared type of the argument.
  ASTType type;
  if (arg.typeExpr) {
    if (arg.variadicKind == VariadicKind::PackVarArg) {
      // Ts in "*args: *Ts" is a reference to a variadic list of types, but
      // needs to be type checked to an instance of VariadicPack.
      type = typeCheckVariadicPack(arg, idx, typeEmitter, tcSignature);
    } else if (arg.variadicKind == VariadicKind::PosVarArg) {
      // "*args: Int" is an instance of VariadicList.
      type = typeCheckVariadicList(arg, typeEmitter, tcSignature);
    } else {
      // Emit the argument type. Allow argument types to be "automatically"
      // parameterized: if the type is fully unbound, its parameters are
      // appended to the function parameters.
      type = typeEmitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);
    }

    // If the type couldn't be emitted, mark this argument erroneous (so uses
    // within the body of the function don't trigger secondary errors) and
    // mark the function erroneous so calls to it won't resolve.  Put in a
    // placeholder type so we can continue type checking.
    if (!type) {
      type = shared.getTypeCheckErrorType();
      arg.isErroneous = true;
      arg.variadicKind = VariadicKind::None; // Don't break invariants.
    }
    type = addImplicitTypeParams(arg.name, type, tcSignature.paramList,
                                 /*append=*/true, arg.loc);
  } else if (idx == 0 && tcSignature.selfType &&
             // FIXME: This is incorrect, the @static_method decorators haven't
             // been applied yet.
             !isStaticMethod) {
    // If this is the 'self' argument in a struct, default the type to Self.
    type = tcSignature.selfType;

    // This can't be variadic.
    if (arg.variadicKind != VariadicKind::None) {
      shared.emitError(arg.loc)
          << "'self' argument must not be variadic; remove '*' before 'self'";
      arg.variadicKind = VariadicKind::None;
      arg.isErroneous = true;
    }

  } else {
    // Otherwise, this is an error.
    shared.emitError(arg.loc, "argument type must be specified")
        << SourceRange(arg.loc, arg.loc);
    type = shared.getTypeCheckErrorType();
    arg.variadicKind = VariadicKind::None;
    arg.isErroneous = true;
  }
  assert(type && "must have an argument type");
  tcSignature.argTypes.push_back(type);

  // Reject function types with unbound parameters that cannot be bound at a
  // call site. Singleton parameters (e.g. origins, including those implied by
  // a variadic pack) are fine: they are removed before elaboration.
  if (auto fType = sugarDynCast<FnTypeGeneratorType>(type)) {
    for (Type paramType : fType.getInputParamTypes()) {
      if (ASTType(paramType).isSingleton(shared))
        continue;
      arg.isErroneous = true;
      auto diag = shared.emitError(arg.typeExpr->getLoc(),
                                   "parametric functions must not be used as "
                                   "arguments; pass as a parameter instead");
      diag.attachNote(arg.typeExpr->getLoc())
          << "alternatively, bind its type parameters to create a concrete "
             "function";
      break;
    }
  }

  // Type check 'deinit' arguments.
  if (arg.convention == ParsedArgument::kConventionDeinit) {
    if (arg.isErroneous) {
      arg.convention = ParsedArgument::kConventionVar;
    } else if (!tcSignature.selfType) {
      shared.emitError(
          arg.loc,
          "'deinit' convention is only valid on struct method arguments");
      arg.convention = ParsedArgument::kConventionVar;
    } else if (!type.getWithoutParameters(shared).isEqualCanon(
                   tcSignature.selfType.getWithoutParameters(shared))) {
      shared.emitError(
          arg.loc, "'deinit' must only be applied to arguments of Self type");
      arg.convention = ParsedArgument::kConventionVar;
    } else if (arg.variadicKind != VariadicKind::None) {
      shared.emitError(arg.loc, "'deinit' arguments must not be variadic");
      arg.convention = ParsedArgument::kConventionVar;
    } else if (tcSignature.fnInfo.kind == SpecialFunctionKind::kInit &&
               tcSignature.argList.parsedArgs.size() == 1) {
      // A lone Self-typed 'deinit' argument gives an initializer
      // move-constructor shape, but only the keyword-only name 'move' is
      // recognized as one (see TypeCheckedFnSignature's constructor); any
      // other spelling is silently shadowed by the synthesized default move
      // constructor. This most commonly bites code written before the
      // argument was renamed from 'take' to 'move'.
      shared.emitWarning(arg.loc, "'deinit' argument '")
          << arg.name.strref()
          << "' does not define a move constructor; declare it as '"
          << SpecialFunctionInfo::get(SpecialFunctionKind::kMoveCtor).name
          << "'";
    }
  }

  // If no convention was explicitly specified, default to 'imm'.
  if (arg.convention == ParsedArgument::kConventionUnspec) {
    // This invariant comes from the parser; other paths, such as a future user
    // IR-rewrite, could violate it, so hard-fail rather than assume it holds.
    if (arg.variadicKind == VariadicKind::KwVarArg)
      llvm::report_fatal_error("kwargs argument has no convention");
    arg.convention = ParsedArgument::kConventionImm;
  }

  // Emit default argument values if present. An error would have been emitted
  // if failed.
  auto defaultVal =
      emitDefault(tcSignature.argList.parsedArgs, idx, type,
                  tcSignature.defaults, typeEmitter, EC_DefaultArgument);
  tcSignature.defaults.push_back(defaultVal);

  // Now that we have the declared type and default value sorted, apply the
  // argument convention to compute the full type for the argument.
  switch (arg.convention) {
  case ParsedArgument::kConventionUnspec:
    llvm_unreachable("should be resolved by now");
  case ParsedArgument::kConventionByRefResult:
    llvm_unreachable("shouldn't occur in an argument list");
  case ParsedArgument::kConventionVar:
    // Owned arguments are always passed in memory, allowing us to check for
    // exclusivity and other requirements.  Register passable arguments are
    // promoted to being passed in registers after elaboration.
    arg.kgenConvention = ArgConvention::OwnedMem;
    break;
  case ParsedArgument::kConventionDeinit:
    arg.kgenConvention = ArgConvention::DeinitMem;
    break;
  case ParsedArgument::kConventionRef: {
    if (arg.variadicKind != VariadicKind::None) {
      // There should be no reason this isn't supportable.
      shared.emitError(
          arg.loc, "TODO: variadic isn't supported with 'ref' convention yet");
      arg.variadicKind = VariadicKind::None;
    }
    auto refType = processRefOriginSpecifier(arg.refOriginExpr, type, arg.name,
                                             tcSignature.paramList,
                                             RefSpecifierKind::kRefArgument);
    type = refType;
    if (refType.isMutableKnown(true))
      arg.kgenConvention = ArgConvention::MutRef;
    else
      arg.kgenConvention = ArgConvention::Ref;

    if (isa<TypeCheckErrorType>(type.getReferenceElementType()))
      arg.isErroneous = true;
    break;
  }
  case ParsedArgument::kConventionImm: {
    arg.kgenConvention = ArgConvention::ImmMem;
    // For parametric closure traits, we don't care about register passability.
    // All arguments will be parsed as if they are mem type, such that we can
    // match `def (T)` with both `def (Int)` and `def (MemType)`.
    if (tcSignature.argList.isExperimentalParamTrait)
      break;

    TypeConvention conv = type.getRegisterPassability(arg.loc, shared);
    // FIXME(MOCO-725): Borrows of non-trivial register-passable values don't
    // have origins and can't be correctly tracked if captured in an async
    // function. Emit an error to avoid a footgun.
    if (arg.variadicKind != VariadicKind::PackVarArg &&
        conv == TypeConvention::RegisterPassable &&
        tcSignature.argList.effects.isAsync()) {
      shared.emitError(
          arg.loc, "TODO: read-only non-trivial register-passable arguments "
                   "are not yet supported in async functions");
    }
    // We can pass trivial register borrowed arguments in a register.  We cannot
    // pass non-trivial ones because we cannot diagnose ownership and have other
    // lifetime issues.
    if (conv == TypeConvention::RegisterPassableTrivial)
      arg.kgenConvention = ArgConvention::ImmReg;
    break;
  }
  case ParsedArgument::kConventionMut:
    arg.kgenConvention = ArgConvention::Mut;
    break;

  case ParsedArgument::kConventionOut:
    llvm_unreachable("Should remove this");
    break;
  }

  // For variadics, we figure out the declared arg convention and adjust passed
  // convention.
  if (arg.variadicKind == VariadicKind::PackVarArg ||
      arg.variadicKind == VariadicKind::PosVarArg) {
    // Remember the original declared convention, forcing to memory convention.
    // The VariadicPack itself is passed as borrowed except for owned
    // convention: this allows the callee to consume the pack.
    switch (arg.convention) {
    case ParsedArgument::kConventionRef:
    case ParsedArgument::kConventionUnspec:
    case ParsedArgument::kConventionByRefResult:
    case ParsedArgument::kConventionOut:
    case ParsedArgument::kConventionDeinit:
      llvm_unreachable("not a variadic arg convention");
    case ParsedArgument::kConventionVar:
      arg.variadicArgConvention = ArgConvention::OwnedMem;
      arg.kgenConvention = ArgConvention::OwnedMem;
      break;
    case ParsedArgument::kConventionImm:
      arg.variadicArgConvention = ArgConvention::ImmMem;
      arg.kgenConvention = ArgConvention::ImmMem;
      break;
    case ParsedArgument::kConventionMut:
      arg.variadicArgConvention = ArgConvention::Mut;
      arg.kgenConvention = ArgConvention::ImmMem;
      break;
    }
  }

  // Values passed by memory need an associated origin parameter, and need to
  // be passed by reference. For now, we don't use reference types in **kwargs.
  Type fullType;
  if (hasImplicitOrigin(arg.kgenConvention) &&
      arg.variadicKind != VariadicKind::KwVarArg) {
    bool isMutable = arg.kgenConvention != ArgConvention::ImmMem;
    fullType =
        makeImplicitRefTypeForArg(arg, idx, type, isMutable, tcSignature);
  } else {
    fullType = type;
  }

  // More special cases for kwVarArg's.
  if (arg.variadicKind == VariadicKind::KwVarArg) {
    // We build StringDict[ValType].
    ASTType dictType = shared.getStandardCollectionType(arg.loc, "StringDict");

    auto dictDecl = cast<LIT::StructType>(dictType.mlirType);
    // We know these are all UnboundAttrs created by
    // StructDeclOp::bindReference. The correct way is to have bindReference
    // return a GeneratorType.
    ArrayRef<TypedAttr> inputUnboundParams = dictDecl.getParamValues();
    if (inputUnboundParams.size() != 1) {
      shared.emitError(arg.loc)
          << "internal compiler error: StringDict type has unexpected "
             "parameter signature; please file a bug";
      arg.isErroneous = true;
    }

    // If anything is wrong with the argument, we terminate before emitting a
    // type for the variadic keyword arguments.
    if (arg.isErroneous)
      return;

    auto collectionElement = cast<TraitType>(inputUnboundParams[0].getType());
    SyntheticNode typeExpr(arg.loc);
    auto typeExprToUse = arg.typeExpr ? arg.typeExpr : &typeExpr;
    auto binding = typeEmitter.emitPValue({fullType, typeExprToUse}, EC_Type,
                                          collectionElement);
    if (!binding) {
      arg.isErroneous = true;
      return;
    }
    fullType = cast<LIT::StructType>(dictType).bindAll(binding.get());

    // StringDict is memory only and since only the callee can access it,
    // we pass it as owned.
    arg.kgenConvention = ArgConvention::OwnedMem;
    fullType = makeImplicitRefTypeForArg(arg, idx, fullType, /*isMutable*/ true,
                                         tcSignature);
  }
  tcSignature.fullArgTypes.push_back(fullType);

  // Add the declaration for the argument, now that is has been resolved. Use
  // a placeholder value to allow the value to be referenced, but in function
  // body resolution, it will be replaced with the actual function argument
  // SSA value.
  //
  // Names are always present for function bodies, but can be missing in
  // function types.  In that case, there are obviously no dependent values on
  // it, because they can't be named.
  if (arg.name.empty())
    return;

  // Create the block argument that will eventually represent this function
  // argument.  If we're generating this argument for a function, put it into
  // its entry block. Otherwise it is a function type: We allocate the argument
  // into a holding block owned by SharedState so it isn't leaked.
  Block &blockOwningArg =
      fnDecl ? *fnOp.getBody() : shared.getArgumentOwningBlock();
  BlockArgument bbArg =
      blockOwningArg.addArgument(fullType, shared.translateLocation(arg.loc));

  DeclIRValue argIRValue;
  if (arg.kgenConvention == ArgConvention::ImmReg)
    argIRValue = SRValue(bbArg);
  else // Everything else is passed in memory.
    argIRValue = CValue::getMValueForRef(bbArg);

  ASTDecl &decl = typeEmitter.getDeclResolver().addFullyResolvedDecl(
      argIRValue, arg.name, arg.loc, &typeEmitter.declScope);

  // If we don't have a function decl, notify the listener immediately (function
  // arguments will be notified when they are fully resolved later).
  if (!fnDecl)
    shared.notifyListenerOnArgumentDecl(decl, arg.name, arg.loc);
}

/// Type check the result type for the function.  `resultTypeExpr` will be
/// non-null if explicitly specified in source code, and the `resultLoc` will
/// always be valid point for end of the argument list.
static void typeCheckResult(ParsedArgument resultArg, ASTDecl *fnDecl,
                            TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = tcSignature.paramList.shared;

  // Determine the result type based on what was explicitly written or what
  // the right implicit result type is.
  ASTType resultType;
  if (resultArg.typeExpr &&
      resultArg.typeExpr->kind == ExprNode::kNoneLiteral) {
    // If the result type is a `None` literal, then convert it to NoneType.
    resultType = shared.getNoneType();
  } else if (resultArg.typeExpr) {
    IREmitter typeEmitter(declScope, EC_Type,
                          &tcSignature.paramList.deferredTypingContext);
    resultType = typeEmitter.emitExprType(resultArg.typeExpr);

    // If the resultType is a generated type, ensure that it is convertible to a
    // concrete type as we do not allow runtime generators, and we do not
    // auto-parameterize on result type.
    if (auto paramType = sugarDynCastIfPresent<ParamType>(resultType)) {
      if (auto genType =
              sugarDynCast<GeneratorType>(paramType.getParam().getType())) {
        if (!genType.getInputParamTypes().empty()) {
          shared.emitError(resultArg.typeExpr->getLoc())
              << "result type cannot be parametric";
          resultType = shared.getTypeCheckErrorType();
        } else {
          // If the generator type has no input parameters, we're good as long
          // as any constraints on the generator can be discharged by the
          // function's constraints.
          for (ConstraintAttr c : genType.getBodyConstraints())
            tcSignature.paramList.deferredTypingContext.deferredConstraints
                .push_back({c, resultArg.typeExpr->getLoc()});
          llvm::BitVector discharged(genType.getBodyConstraints().size(), true);
          // Record the type as the discharged body type.
          resultType = BindParamsAttr::get(
              shared.getContext(), paramType.getParam(), /*paramValues=*/{},
              /*discharged=*/
              getDenseBoolArrayAttr(shared.getContext(), discharged),
              &shared.getEvaluationContext());
        }
      }
    }

    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType)
      resultType = shared.getTypeCheckErrorType();
  } else if (tcSignature.fnInfo.isInitializer() &&
             resultArg.convention == ParsedArgument::kConventionOut) {
    // If this is an initializer with an 'out self' argument, infer Self.
    resultType = tcSignature.selfType;
  } else {
    // If the result type wasn't specified, we default to "None".
    resultType = shared.getNoneType();
  }

  // If a result origin is specified with `ref [life] Ty`, then form a ref
  // result.
  if (resultArg.convention == ParsedArgument::kConventionRef) {
    if (tcSignature.argList.effects.isAsync()) {
      // TODO(MOCO-787): Async functions don't support ref results yet. We need
      // to define a `CoroutineRef` or support perfect forwarding in generic
      // results.
      shared.emitError(resultArg.refOriginExpr->getLoc())
          << "TODO: ref results aren't supported in async functions yet";
      resultArg.refOriginExpr = nullptr;
    } else {
      resultType = processRefOriginSpecifier(
          resultArg.refOriginExpr, resultType,
          // TODO: Use the name of the return slot if present.
          "__result__", tcSignature.paramList, RefSpecifierKind::kRefResult);
      tcSignature.argList.effects.setRefResult(isa<RefType>(resultType));
    }
  }

  // Remember the user-declared result type.
  tcSignature.resultType = resultType;

  // Process any origins in the result type.
  if (auto resultOrigins =
          shared.cachedOriginFinder.findOriginsIn({resultType});
      !resultOrigins.empty()) {

    // Function result types containing an interior origin establish those
    // origins when called.
    //
    // TODO: Instead of treating this as binary, we should actually capture
    // which indirect origins are being defined.  A binary flag is not the right
    // answer, because parametric types can be substituted in that contain
    // indirect origins, but should be sufficient for now.
    for (TypedAttr origin : resultOrigins)
      origin.walk([&](Attribute nested) {
        if (isa<InteriorOriginAttr, OriginSubtreeAttr>(nested))
          tcSignature.definesInteriorOrigins = true;
      });

    // Check to see if the result type has any embedded origins that refer to
    // in-memory argument origins of generic type, e.g.:
    //
    //     def get[T: AnyType](a: T) -> Pointer[T, origin_of(a)]:
    //        return Pointer(a)
    //
    // These origins are not allowed to be returned from the function, because
    // when instantiated with a register-passable type, argument convention
    // lowering will turn them into:
    //
    //     def get[T: AnyType](borrow_in_reg a: T)
    //                             -> Pointer[T, origin_of(tmp)]:
    //        var tmp = a
    //        return Reference(tmp)
    //
    // Note that we're now returning a reference to something that doesn't
    // outlast the function!
    SmallDenseMap<TypedAttr, size_t, 8> possiblyRegisterPassableOrigins;
    for (auto [idx, parsedArg, fullType] : llvm::enumerate(
             tcSignature.argList.parsedArgs, tcSignature.fullArgTypes)) {

      // Only look at mut, read, owned arguments.  RegisterPassable args
      // won't have a origin, and `ref` args are not lowered by-reg.
      if (!hasAddress(parsedArg.kgenConvention) ||
          parsedArg.kgenConvention == ArgConvention::Ref ||
          parsedArg.kgenConvention == ArgConvention::MutRef)
        continue;

      // The argument is only a potential problem if it is generic that might
      // expand to a RegisterPassable type.
      auto refType = cast<RefType>(fullType);
      if (!ASTType(refType.getElementType())
               .mightBeRegisterPassable(parsedArg.loc, shared))
        continue;

      // Ok, this origin is a problem.
      possiblyRegisterPassableOrigins[refType.getOrigin()] = idx;
    }

    // Now that we know all the problematic origins, check to see if any of
    // them are referenced.
    for (TypedAttr origin : resultOrigins) {
      // Don't allow mutability dropping to interfere.
      origin = OriginMutCastAttr::strip(origin);
      if (!possiblyRegisterPassableOrigins.count(origin))
        continue;

      // Oops, found a problem, report it and indicate the argument at fault.
      assert(resultArg.typeExpr && "implicit result types can't have origins");
      size_t argIdx = possiblyRegisterPassableOrigins[origin];
      const ParsedArgument &badArg = tcSignature.argList.parsedArgs[argIdx];
      auto diag = shared.emitError(resultArg.typeExpr->getLoc());
      diag << "cannot return " << badArg.name << "s origin, because it ";
      ASTType argType =
          ASTType(tcSignature.fullArgTypes[argIdx]).getReferenceElementType();
      if (argType.isRegisterPassable(badArg.loc, shared))
        diag << "has RegisterPassable type " << argType;
      else
        diag << "might expand to a RegisterPassable type";
      diag << resultArg.typeExpr->getRange()
           << SourceRange(badArg.loc, badArg.loc);
      break;
    }
  }

  // Now that we have the user's result type, compute the full type of the
  // result, which can can be different when memory only, when throwing, etc.
  ASTType fullResultType = resultType;
  TypeConvention rp = resultType.getRegisterPassability(resultArg.loc, shared);

  // If this function throws, add a result slot for the error that may be
  // raised.
  if (tcSignature.argList.effects.isThrows()) {
    ASTType errorType;
    if (tcSignature.argList.thrownTypeExpr) {
      IREmitter typeEmitter(declScope, EC_Type,
                            &tcSignature.paramList.deferredTypingContext);
      errorType = typeEmitter.emitExprType(tcSignature.argList.thrownTypeExpr);
    }
    if (!errorType)
      errorType = shared.lookupBuiltinType(
          "Error", tcSignature.paramList.declScope, resultArg.loc);

    // Synthesize a ByRefError argument for the error.
    ParsedArgument errArg;
    errArg.loc = resultArg.loc;
    errArg.name = StringAttr::get(shared.getContext(), "__error__");
    errArg.convention = ParsedArgument::kConventionByRefResult;
    errArg.kgenConvention = ArgConvention::ByRefError;
    errArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    errArg.typeExpr = tcSignature.argList.thrownTypeExpr;
    tcSignature.argList.parsedArgs.push_back(errArg);
    tcSignature.argTypes.push_back(errorType);
    tcSignature.defaults.push_back(TypedAttr());

    RefType refType = makeImplicitRefTypeForArg(
        errArg, 0, errorType, /*isMutable*/ true, tcSignature);
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.fn declaration (as opposed to a function type),
    // add a block argument for this.
    if (fnDecl) {
      Block &body = *cast<FnOp>(*fnDecl->getIfOperation()).getBody();
      (void)body.addArgument(refType, shared.translateLocation(resultArg.loc));
    }

    // The ABI result type is an scalar<bool> indicating the error state.
    fullResultType = SIMDType::getScalarBoolType(shared.getContext());
    // The result value is always returned through memory.
    rp = TypeConvention::MemoryOnly;
  }

  // Async functions always use in-memory results.
  if (tcSignature.argList.effects.isAsync())
    rp = TypeConvention::MemoryOnly;

  // "out" arguments with specified origins or address spaces use a result slot.
  if (resultArg.refOriginExpr &&
      resultArg.convention == ParsedArgument::kConventionOut)
    rp = TypeConvention::MemoryOnly;

  // If it is memory-only, pass it indirectly as the last argument to the
  // function by-reference.
  if (rp == TypeConvention::MemoryOnly) {
    // Synthesize a ByRefResult argument for the result.
    if (!resultArg.name)
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");

    // Compute the RefType for this new argument with an implicit origin or
    // from the specified ref specification if present.
    RefType refType;
    if (resultArg.refOriginExpr &&
        resultArg.convention == ParsedArgument::kConventionOut) {
      refType = processRefOriginSpecifier(
          resultArg.refOriginExpr, resultType, resultArg.name.strref(),
          tcSignature.paramList, RefSpecifierKind::kOutArgument);
    } else {
      refType = makeImplicitRefTypeForArg(resultArg, 0, resultType,
                                          /*isMutable*/ true, tcSignature);
    }

    resultArg.convention = ParsedArgument::kConventionByRefResult;
    resultArg.kgenConvention = ArgConvention::ByRefResult;
    resultArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    tcSignature.argList.parsedArgs.push_back(resultArg);
    tcSignature.argTypes.push_back(resultType);
    tcSignature.defaults.push_back(TypedAttr());
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.fn declaration (as opposed to a function type),
    // add a block argument for this.  We don't register this for name lookup
    // though, we don't want it to conflict with user identifiers, and it is
    // never looked up directly.
    if (fnDecl) {
      Block &body = *cast<FnOp>(*fnDecl->getIfOperation()).getBody();
      auto bbArg =
          body.addArgument(refType, shared.translateLocation(resultArg.loc));

      // Add a decl so this will be found by name lookup within the body.
      shared.getDeclResolver().addFullyResolvedDecl(
          MLValue(bbArg), resultArg.name, resultArg.loc, &declScope);
    }

    // We know the ABI register result will be None now, which is trivial.
    if (!tcSignature.argList.effects.isThrows())
      fullResultType = shared.getNoneType();
  }

  tcSignature.fullResultType = fullResultType;
}

/// Emit the argument types, default values, and result type and determine
/// the argument conventions.
///
/// 'fnDecl' will be null when this is a function type, which doesn't have a
/// declaration.
TypeCheckedFnSignature::TypeCheckedFnSignature(TypeCheckedParamList &paramList,
                                               ParsedArgumentList &argList,
                                               const ExprNode *originExpr,
                                               ASTDecl *fnDecl,
                                               StringAttr baseName)
    : paramList(paramList), argList(argList) {

  if (baseName) { // Function pointer types don't have a name.
    fnInfo =
        SpecialFunctionInfo::get(SpecialFunctionInfo::lookupKind(baseName));

    // Recognize copy and move constructors by their keyword arg.
    if (fnInfo.kind == SpecialFunctionKind::kInit &&
        argList.parsedArgs.size() == 1 &&
        argList.parsedArgs[0].kwArgHandling == KWArgHandling::kKeywordOnly) {
      if (argList.parsedArgs[0].name.strref() == "move")
        fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kMoveCtor);
      else if (argList.parsedArgs[0].name.strref() == "copy")
        fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kCopyCtor);
    }
  }
  SharedState &shared = paramList.shared;
  IREmitter typeEmitter(paramList.declScope, EC_Type,
                        &paramList.deferredTypingContext);

  // If this definition is a struct/class member, compute the self type.
  if (fnDecl) {
    if (ASTDecl *parent = fnDecl->tryGetMethodParentDecl()) {
      // The parent decl must be fully resolved in order to resolve any of its
      // members.
      assert(parent->resolvedness == DeclResolvedness::body);
      selfType = parent->getTypeDeclSelf();
    }
  }

  // If this is a well-known function like `__init__`, perform early semantic
  // checks and clarify what special function it really is.
  // This logic happens before type checking, so we need to be very careful
  // to only process it if defined correctly.  We let downstream checks diagnose
  // the errors.
  auto checkInitializer = [&]() -> LogicalResult {
    if (!selfType) {
      fnDecl->setErroneous();
      shared.emitError(fnDecl->getLoc(), "'")
          << fnInfo.name << "' must be a method";
      return failure();
    }

    // Initializers without an out argument or a -> Self result are incorrect.
    if (!argList.resultArg.name &&
        (!argList.resultArg.typeExpr || // Allow "no ->" and "-> None"
         argList.resultArg.typeExpr->kind == ExprNode::kNoneLiteral)) {
      shared.emitError(
          fnDecl->getLoc(),
          "__init__ method must return Self type with 'out' argument");
      return failure();
    }

    // TODO(MOCO-789): Async initializers require a `byref_result` thunk to be
    // emitted. Just forbid them for now.
    if (argList.effects.isAsync()) {
      shared.emitError(fnDecl->getLoc())
          << "TODO: async constructors are not yet supported";
      argList.effects.setAsync(false);
      return failure();
    }

    // RegisterPassable values are movable by passing the register around, so
    // they can't define a move ctor.
    if (fnInfo.kind == SpecialFunctionKind::kMoveCtor &&
        selfType.isRegisterPassable(fnDecl->getLoc(), shared)) {
      shared.emitError(fnDecl->getLoc())
          << "'RegisterPassable' types must not declare explicit move "
             "constructors; values of these types have no identity, and "
             "the compiler can freely move them between registers";
      return failure();
    }

    // Trivial types are copyable with memcpy so they can't define copy ctor.
    if (fnInfo.kind == SpecialFunctionKind::kCopyCtor &&
        selfType.isTrivial(fnDecl->getLoc(), shared)) {
      shared.emitError(fnDecl->getLoc())
          << "trivial types must not declare explicit copy constructors; "
             "they are trivially copyable";
      return failure();
    }

    return success();
  };

  // Check initializers for validity.
  if (fnInfo.isInitializer()) {
    if (failed(checkInitializer())) {
      fnDecl->setErroneous();
      fnInfo = SpecialFunctionInfo();
    }
  }

  // __new__ and __init__ are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    cast<FnOp>(*fnDecl->getIfOperation()).setIsStatic(true);

  if (fnInfo.isInstMethod() && !selfType) {
    shared.emitError(fnDecl->getLoc())
        << "'" << baseName.getValue()
        << "' must be a method, not a global function";
    fnDecl->setErroneous();
    fnInfo = SpecialFunctionInfo();
  }

  // Trivial types are copyable with memcpy so they can't define a dtor.
  if (fnInfo.kind == SpecialFunctionKind::kDeinit &&
      selfType.isTrivial(fnDecl->getLoc(), shared)) {
    fnDecl->setErroneous();
    auto diag =
        shared.emitError(fnDecl->getLoc(), "trivial types must not declare '");
    diag << baseName.getValue() << "' methods; they are trivially destroyable";
    diag.attachNote(*fnDecl)
        << "trivial types have no identity; the compiler destroys them "
           "automatically with no observable effect";
    fnInfo = SpecialFunctionInfo();
  }

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  for (size_t i = 0, e = argList.parsedArgs.size(); i != e; ++i)
    typeCheckOneArgument(i, fnDecl, *this);

  // Compute the result type.
  typeCheckResult(argList.resultArg, fnDecl, *this);

  // If a capture origin set was specified, emit it. It will be added to the
  // signature type later.
  if (originExpr) {
    // Special rule for `[_]` when specifying the capture origin set: the
    // set is unbound and will be autoparameterized.
    auto setType = OriginSetType::get(shared.getContext());
    if (originExpr->kind == ExprNode::kDiscardLiteral) {
      captureOrigins = UnboundAttr::get(setType);
    } else {
      captureOrigins =
          typeEmitter.emitExprPValue(originExpr, EC_Origin, setType);
    }
  }

  // Type check & emit the constraints.
  paramList.emitBodyConstraints();
}

/// This performs any special checks over the declaration based on its name
/// and whether it is a method.  This happens after decorator processing
/// because that is how defs work in Python.
///
/// If this function detects a problem, it marks the decl as erroneous and
/// resets fnInfo.
void TypeCheckedFnSignature::verifyFunctionNameBinding(ASTDecl &decl,
                                                       StringAttr &name) {
  FnOp funcOp = cast<FnOp>(*decl.getIfOperation());

  MutableArrayRef<ParsedArgument> parsedArgs = argList.parsedArgs;
  ArrayRef<Type> argTypes = this->argTypes;
  auto &shared = paramList.shared;

  // '__del__' is a deprecated spelling of the destructor; canonicalize it to
  // '__deinit__' here so every downstream consumer of the function's name
  // (the AST and the MLIR it lowers to) only ever sees the canonical form.
  if (fnInfo.kind == SpecialFunctionKind::kDeinit &&
      name.getValue() == "__del__") {
    shared.emitWarning(decl.getLoc(),
                       "'__del__' is deprecated; use '__deinit__'")
        << FixIt::replaceToken(decl.getLoc(), "__deinit__");
    name = StringAttr::get(shared.getContext(), "__deinit__");
  }

  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc,
                          const Twine &message = Twine()) -> MojoInflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message = Twine()) -> MojoInflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(funcOp.getLoc(), message);
  };

  // If the argument list has a mut result or mut error, ignore it for type
  // checking purposes.
  while (!parsedArgs.empty() && parsedArgs.back().convention ==
                                    ParsedArgument::kConventionByRefResult) {
    parsedArgs = parsedArgs.drop_back();
    argTypes = argTypes.drop_back();
  }

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  constexpr size_t kSelfArgNo = 0;
  if (ASTDecl *parent = decl.getParentDecl();
      parent &&
      isa_and_nonnull<StructDeclOp, TraitDeclOp>(parent->getIfOperation())) {
    // The parent decl must be fully resolved in order to resolve any of its
    // members.
    assert(parent->resolvedness == DeclResolvedness::body);
    selfType = parent->getTypeDeclSelf();
  }

  // Check any special function information.

  // Check that the 'self' argument/result of a method was specified correctly.
  if (selfType) {
    if (fnInfo.flags & SpecialFunctionInfo::kSelfResult) {
      // __new__ and __init__ require a Self result type, or a specialization
      // thereof.
      checkSelfArgument(decl, resultType, argList.resultArg,
                        /*isSelfResult*/ true);
    } else if (funcOp.getIsStatic()) {
      // Static methods don't have a self argument.
    } else if (argTypes.empty()) {
      emitError("self argument must be present in instance method");
    } else {
      // Normal methods require a self argument.
      checkSelfArgument(decl, argTypes[kSelfArgNo], parsedArgs[kSelfArgNo],
                        /*isSelfResult*/ false);
    }
  }

  // Verify the argument count lines up.
  if (fnInfo.kind != SpecialFunctionKind::kNormal) {
    size_t numActualArgs = parsedArgs.size();
    size_t numMin = fnInfo.minNumArguments;
    ssize_t numMax = fnInfo.maxNumArguments;
    if (numMin == size_t(numMax) && numActualArgs != numMin) {
      emitError() << name << " requires " << numMin << " operand"
                  << plural(numMin);
    } else if (numActualArgs < numMin) {
      emitError() << name << " requires at least " << numMin << " operand"
                  << plural(numMin);
    } else if (numMax != -1 && numActualArgs > size_t(numMax)) {
      emitError() << name << " requires at most " << size_t(numMax)
                  << " operand" << plural(numMax);
    }
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType) {
      emitError() << name << " must be a method";
    } else if (funcOp.getIsStatic()) {
      if (!(fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod))
        emitError("special method must not be a static method; remove the "
                  "static method decorator");
    }
  }

  // Get the user-declared result type, which might be a memory-only type.
  ASTType declaredResultType = resultType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() && !declaredResultType.isNoneType())
    emitError() << name << " result type must be elided (or None)";

  // Reject special functions declared as throwing when that is invalid.
  if (argList.effects.isThrows() &&
      fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
    const char *fnName;
    if (fnInfo.kind == SpecialFunctionKind::kCopyCtor)
      fnName = "copy constructor";
    else if (fnInfo.kind == SpecialFunctionKind::kMoveCtor)
      fnName = "move constructor";
    else {
      assert(fnInfo.kind == SpecialFunctionKind::kDeinit);
      fnName = "destructor";
    }

    emitError() << fnName
                << " must not declare 'raises'; remove the 'raises' keyword";
  }

  // Shared logic to diagnose the 'self' argument of __del__ and __moveinit__.
  auto diagnoseSelfForDelAndMoveInit = [&](const char *argName) {
    // This method is going to consume the passed value.
    if (parsedArgs[kSelfArgNo].convention != ParsedArgument::kConventionVar &&
        parsedArgs[kSelfArgNo].convention !=
            ParsedArgument::kConventionDeinit) {
      emitErrorLoc(parsedArgs[kSelfArgNo].loc, "'")
          << argName << "' argument must be passed as 'deinit'";
      parsedArgs[kSelfArgNo].convention = ParsedArgument::kConventionDeinit;
    }
  };

  // Diagnose common errors and handle other special cases.
  switch (fnInfo.kind) {
  default:
    break;
  case SpecialFunctionKind::kNew:
    emitError("'__new__' is not supported on structs; use '__init__' instead");
    break;
  case SpecialFunctionKind::kMLIRI1:
    if (!declaredResultType.mlirType.isSignlessInteger(1))
      emitError() << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kCopyCtor: {
    assert(parsedArgs.size() == 1 && "arg count already checked above");
    if (parsedArgs[kSelfArgNo].convention == ParsedArgument::kConventionImm) {
      // ok.
    } else if (parsedArgs[kSelfArgNo].convention ==
               ParsedArgument::kConventionRef) {
      // Allow ref if the origin is immutable or flexible.
      if (cast<RefType>(fullArgTypes[kSelfArgNo]).isMutableKnown(true)) {
        emitErrorLoc(
            parsedArgs[kSelfArgNo].loc,
            "existing value argument must be passed as an immutable 'ref'");
      }
    } else {
      emitErrorLoc(parsedArgs[kSelfArgNo].loc,
                   "existing value argument must be passed as 'imm'");
    }
    break;
  }
  case SpecialFunctionKind::kMoveCtor: {
    assert(parsedArgs.size() == 1 && "arg count already checked above");
    diagnoseSelfForDelAndMoveInit("move");
    break;
  }
  case SpecialFunctionKind::kDeinit:
    assert(parsedArgs.size() == 1 && "arg count already checked above");
    diagnoseSelfForDelAndMoveInit("self");
    break;
  }

  // If we have a special function kind and didn't have any errors with it,
  // remember which kind it is.
  if (fnInfo.kind != SpecialFunctionKind::kNormal)
    funcOp.setSpecialFnKind(uint8_t(fnInfo.kind));
}

/// In a method with a self argument, check to make sure it has the correct
/// type and invariants. "isSelfResult" is true for initializers that return
/// Self, they don't take it as an input argument.
void TypeCheckedFnSignature::checkSelfArgument(ASTDecl &decl,
                                               ASTType selfArgType,
                                               const ParsedArgument &selfArg,
                                               bool isSelfResult) const {
  auto &shared = paramList.shared;

  // Don't check broken args, because we don't want redundant diagnostics.
  if (selfArg.isErroneous)
    return;
  // It ok if it exactly matches.
  if (selfType.isEqualCanon(selfArgType))
    return;

  // If an error was already diagnosed with the type, disable follow-ons.
  if (isa<TypeCheckErrorType>(selfArgType)) {
    selfArg.isErroneous = true;
    return;
  }

  auto emitErrorLoc = [&](SMLoc loc) -> MojoInflightDiag {
    decl.setErroneous();
    return shared.emitError(loc);
  };

  if (!isSelfResult && !allowCustomSelfType) {
    emitErrorLoc(selfArg.loc)
        << "'self' argument must have type 'Self'; use a 'where' clause "
           "to constrain the 'Self' type instead";
    return;
  }

  // It is ok if the self type has different parameters than the
  // declaration, this is a form of conditional conformance.
  if (selfType.getDecl(shared) != selfArgType.getDecl(shared)) {
    // Otherwise, this is an unrecognized self type. If this is a trait,
    // the explicit self type is very hard to specify in mojo, so we
    // suggest to use 'Self' instead.
    auto diag = emitErrorLoc(selfArg.loc) << "'self' argument must have type ";
    if (decl.getParentDecl() &&
        isa_and_nonnull<TraitDeclOp>(decl.getParentDecl()->getIfOperation()))
      diag << "'Self' in trait method declaration";
    else
      diag << selfType;
    diag << ", but actually has type " << selfArgType;
    if (selfArg.typeExpr)
      diag << selfArg.typeExpr->getRange();
    selfArg.isErroneous = true;
    return;
  }

  // We are done with normal methods here.
  if (!isSelfResult)
    return;

  // If this is an init with a specialized result that has different
  // parameters than Self, then the declared parameter cannot be used - such
  // a thing would cause cycles in parameter inference that cannot solve:
  //   struct Cyclic[a: Int]: def __init__(out self: Cyclic[Self.a + 1]): ...
  // or that can lead to complicated cases like:
  //   struct Weird[T: AnyType]: def __init__(a: T, out self: Weird[Int]):
  // if you can compute the result type, you don't need to also use it some
  // other way.
  ArrayRef<TypedAttr> selfParams = selfType.getParamBindings();
  ArrayRef<TypedAttr> selfResultParams = selfArgType.getParamBindings();
  assert(selfParams.size() == selfResultParams.size() &&
         "self params and result params must match");
  SmallDenseMap<ParamDeclRefAttr, TypedAttr, 2> disabledParams;
  for (auto [selfParam, resultParam] :
       llvm::zip(selfParams, selfResultParams)) {
    // Don't disable the parameter if it is the same as the result, e.g.
    // disable 'x' but not 'y' here:
    //   struct T[x: Int, y: Int]:
    //     def __init__(out self: T[1, y]): ...
    if (selfParam != resultParam)
      disabledParams[cast<ParamDeclRefAttr>(selfParam)] = resultParam;
  }

  // If any parameters are disabled, scan the parameter and argument list
  // to make sure they didn't appear there.
  if (disabledParams.empty())
    return;

  bool hadError = false;
  SmallVector<ParamDeclRefAttr> paramUses;
  auto checkDecl = [&](Type argType, SMLoc loc, const ExprNode *expr) {
    paramUses.clear();
    shared.collectParamRefsInType(argType, paramUses);
    for (ParamDeclRefAttr use : paramUses) {
      if (!disabledParams.count(use))
        continue;
      auto diag = emitErrorLoc(loc)
                  << "cannot use Self parameter " << use
                  << " in constructor whose result defines it to "
                  << disabledParams[use];
      if (expr)
        diag << expr->getRange();
      selfArg.isErroneous = true;
      hadError = true;
      return;
    }
  };

  // Check the parameters.
  for (auto [param, paramDeclAttr, paramLoc] :
       llvm::zip(paramList.paramDeclAttrs, paramList.paramDeclAttrs,
                 paramList.locations)) {
    checkDecl(paramDeclAttr.getType(), paramLoc, nullptr);
    if (hadError)
      return;
  }

  // Check the arguments.
  for (auto [arg, argType] : llvm::zip(argList.parsedArgs, argTypes)) {
    checkDecl(argType, arg.loc, arg.typeExpr);
    if (hadError)
      return;
  }
}

FunctionType TypeCheckedFnSignature::getFunctionType() const {
  return FunctionType::get(fullResultType.mlirType.getContext(), fullArgTypes,
                           {fullResultType.mlirType});
}

/// Form a LIT signature packaging up all the stuff we need to know about this
/// type checked function.
FnTypeGeneratorType TypeCheckedFnSignature::getFnTypeGeneratorType() const {
  MLIRContext *ctx = paramList.shared.getContext();

  size_t numArgs = argList.parsedArgs.size();
  SmallVector<PogMetadataAttr> argPogs;
  argPogs.reserve(numArgs);
  SmallVector<ArgConvention> argConventions;
  argConventions.reserve(numArgs);

  ArgConvention argVariadicOrigConvention = ArgConvention::ByRefError;
  [[maybe_unused]] int numVariadics = 0;
  for (auto [idx, arg] : llvm::enumerate(argList.parsedArgs)) {
    if (arg.variadicKind == VariadicKind::PosVarArg ||
        arg.variadicKind == VariadicKind::PackVarArg) {
      argVariadicOrigConvention = arg.variadicArgConvention;
      ++numVariadics;
    }

    argPogs.emplace_back(
        PogMetadataAttr::get(arg.name, arg.getKWArgHandlingAsPassingKind(),
                             arg.variadicKind, defaults[idx]));
    argConventions.push_back(arg.kgenConvention);
  }
  assert(numVariadics <= 1 && "There can be at most one variadic argument");

  PogListAttr paramListAttr = paramList.getParamListAttr();
  auto argListAttrs = PogListAttr::get(ctx, argPogs, /*bodyConstraints=*/{},
                                       argVariadicOrigConvention);
  auto metadata = FnMetaOriginDataAttr::get(
      ctx, implicitOriginDecls.size(),
      getOriginsAccessibleByParams(paramListAttr, paramList.paramDeclAttrs,
                                   paramList.shared, captureOrigins),
      isNestedOriginsReadOnly, definesInteriorOrigins);

  /// We shouldn't have internal verifier errors here, but if we do, try to
  /// emit them at some location of merit because something snuck through
  /// semantic analysis.  We should treat these as bugs in the parser.
  auto emitError = [&] {
    // Find some loc to use.
    Location loc = UnknownLoc::get(ctx);
    if (!argList.parsedArgs.empty())
      loc = paramList.shared.translateLocation(argList.parsedArgs[0].loc);
    else if (!paramList.locations.empty())
      loc = paramList.shared.translateLocation(paramList.locations.back());
    return mlir::emitError(loc);
  };

  FunctionType functionType = getFunctionType();
  return FuncTypeGeneratorType::remapToFuncTypeGenerator(
      paramList.paramDeclAttrs, functionType, argConventions, argList.effects,
      metadata, paramListAttr, emitError, argListAttrs);
}
