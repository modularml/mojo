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
// This file provides the implementation of the AST printing logic.
//
//===----------------------------------------------------------------------===//

#include "IREmitter.h"
#include "Mojo/lib/MojoParser/Traits.h"
#include "ParserEvaluationContext.h"

#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPAttrs.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a parameter value of MLIR wrapper type like Bool or Int or DType,
/// dig out the single element of the struct with the specified type.
template <typename T>
static T getSingleElementStructAttr(TypedAttr param) {
  if (auto strParam = sugarDynCast<LITStructAttr>(param)) {
    if (strParam.getValues().size() == 1)
      return sugarDynCast<T>(std::get<1>(strParam.getValues()[0]));
  }
  return {};
}

/// Given a SymbolRefAttr, return the underlying symbol name.
static StringRef getNameFromSymbolRef(SymbolRefAttr symbol) {
  StringAttr leaf;
  if (symbol.getNestedReferences().empty())
    leaf = symbol.getRootReference();
  else
    leaf = symbol.getNestedReferences().back().getAttr();

  // Demangle the name.  We can end up with functions like:
  //   "__init__[LITImmOrigin,::Origin[::Bool(False), $0]](::Int*)"
  // and structs like:
  //   ParamStruct[:trait<_\"std::builtin::value::TrivialRegisterPassable\">
  //   _\"std::builtin::int::Int\"])
  StringRef name = leaf.getValue();
  if (size_t mangleStart = name.find_first_of("[(");
      mangleStart != std::string::npos)
    name = name.take_front(mangleStart);
  return name;
}

/// Try to extract a symbol reference and parameter list from a function callee.
/// Returns the symbol being called and the parameters, or a null symbol if
/// decoding failed.
static std::pair<SymbolRefAttr, ArrayRef<TypedAttr>>
tryGetSymbolNameAndParams(TypedAttr param) {
  param = ParamOperatorAttr::stripRebind(param);
  if (auto symbolCst = sugarDynCast<SymbolConstantAttr>(param))
    return {symbolCst.getSymbol(), symbolCst.getParamValues()};
  return {{}, {}};
}

/// If the value is a call to an implicit constructor of the value's type,
/// remove it, otherwise leave the value alone.
///
/// When shared is specified, this makes sure to only strip implicit
/// constructors, but when it is null it will always strip all constructors.
static void removeImplicitCtorCall(TypedAttr &value, SharedState *shared) {
  // Look through SugarAttr to find the underlying apply if present. We only
  // need to look through at most one SugarAttr, because the sugared side has
  // any nested sugars removed already.
  auto valueWithoutSugar = value;
  if (auto sugar = dyn_cast<SugarAttr>(valueWithoutSugar))
    if (shared || sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
      valueWithoutSugar = sugar.getSugared();

  // Implicit constructors are always calls.
  auto op = sugarDynCast<ParamOperatorAttr>(valueWithoutSugar);
  if (!op ||
      (op.getOpcode() != POC::Apply &&
       op.getOpcode() != POC::ApplyResultSlot) ||
      op.getOperands().size() != 2) // callee and value to convert.
    return;

  auto [nameAttr, calleeParams] =
      tryGetSymbolNameAndParams(op.getOperands()[0]);
  if (!nameAttr)
    return;
  StringRef name = getNameFromSymbolRef(nameAttr);
  if (name != "__init__")
    return;

  if (shared) {
    ASTDecl *decl = shared->getDeclResolver().getDeclForFuncSymbol(nameAttr);
    if (!decl)
      return;
    auto calleeFn = cast<FnOp>(decl->getIfOperation());
    if (!calleeFn.isImplicitConversion())
      return; // If it's not an implicit conversion, don't remove it.
  }

  value = op.getOperands()[1];
}

// Get the name of the enclosing struct from the function symbol reference.
static StringRef tryGetTypeNameFromSymbolRef(SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().size() >= 2)
    return symbol.getNestedReferences().drop_back().back().getValue();
  return {};
}

/// If every element of \p params is a ParamDeclRefAttr that is an
/// auto-parameterization of the same base argument, return the common base
/// argument name. This detects "identity type reconstructions" like
/// TileTensor[x.dtype, x.layout, ...] which are just re-expansions of x's
/// type, allowing us to print just "x" instead.
///
/// This is safe because StructType::getParamValues() always contains an entry
/// for every declared parameter (unbound slots use UnboundAttr), so we never
/// match a partial application — any non-auto-param entry causes early
/// bail-out.
static StringRef getIdentityReconstructionArgName(ArrayRef<TypedAttr> params) {
  if (params.empty())
    return {};
  StringRef commonBase;
  for (TypedAttr p : params) {
    auto declRef = sugarDynCast<ParamDeclRefAttr>(p);
    if (!declRef)
      return {};

    StringRef name = declRef.getName();
    // Use demangleParameterName to strip the uniquing backtick suffix,
    // e.g. "arg.field`42" -> "arg.field". The dot-extraction that follows
    // is specific to identity reconstruction detection.
    StringRef stripped = demangleParameterName(name);
    if (stripped == name)
      return {}; // No backtick suffix means it's not an auto-param.

    // Extract the base argument name (before the first '.').
    size_t dotPos = stripped.find('.');
    if (dotPos == StringRef::npos)
      return {}; // Not in "arg.field" form.
    StringRef base = stripped.take_front(dotPos);

    if (commonBase.empty())
      commonBase = base;
    else if (commonBase != base)
      return {}; // Different base arguments.
  }
  return commonBase;
}

/// If \p typeValue is a TypeParamAttr wrapping a LIT::StructType whose
/// parameters are all auto-params from the same argument AND whose auto-param
/// field names match the struct's declared parameter names, print
/// "argName.memberName" and return true. Otherwise return false and print
/// nothing.
///
/// The field-name check prevents false positives: when a different struct type
/// is constructed using the same argument's auto-params (e.g. Swapped[p.A, p.B]
/// where p is a TwoParam), the auto-param field names ("A", "B") won't match
/// the target struct's declared names ("X", "Y"), so we correctly bail out.
static bool tryPrintIdentityReconstruction(raw_ostream &os, TypedAttr typeValue,
                                           StringRef memberName,
                                           SharedState *diagShared) {
  auto typeAttr = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeAttr)
    return false;
  auto structTy = dyn_cast<LIT::StructType>(typeAttr.getMlirType());
  if (!structTy)
    return false;
  ArrayRef<TypedAttr> params = structTy.getParamValues();
  StringRef argName = getIdentityReconstructionArgName(params);
  if (argName.empty())
    return false;

  // When the struct declaration is resolvable, verify that each auto-param's
  // field name matches the struct's declared parameter name at the same
  // position. This prevents false positives where a different struct type is
  // constructed using the same argument's auto-params (e.g. Swapped[p.A, p.B]
  // where p is a TwoParam). For cross-module types that can't be resolved,
  // fall through to accept the reconstruction — the basic auto-param check is
  // sufficient since cross-module false positives are unlikely in practice.
  if (ASTDecl *decl =
          diagShared->getDeclResolver().getDeclForTypeSymbolIfExists(
              structTy.getValue().getValue())) {
    auto declIntf = dyn_cast<DeclInterface>(decl->getIfOperation());
    if (!declIntf)
      return false;
    ArrayRef<ParamDeclAttr> declaredParams = declIntf.getInputParams();
    if (declaredParams.size() != params.size())
      return false;
    for (size_t i = 0; i < params.size(); ++i) {
      auto declRef = sugarDynCast<ParamDeclRefAttr>(params[i]);
      StringRef fieldName =
          demangleParameterName(declRef.getName(), /*forUser=*/true);
      StringRef declaredName = demangleParameterName(
          declaredParams[i].getName().getValue(), /*forUser=*/true);
      if (fieldName != declaredName)
        return false;
    }
  }

  os << argName << '.' << memberName;
  return true;
}

// If we are a builtin symbol, then just strip everything but the name of the
// type. E.g. Print ::Int instead of std::builtin::int::Int.
static StringRef trimBuiltinNamespace(StringRef nestedSymbolName) {
  // List of common namespace prefixes to trim
  static const StringRef commonPrefixes[] = {
      "std::", "layout::"
      // Add other common prefixes here
  };

  StringRef prettyName(nestedSymbolName);
  for (StringRef prefix : commonPrefixes) {
    if (prettyName.starts_with(prefix)) {
      const size_t lastSeparatorLoc = prettyName.rfind("::");
      if (lastSeparatorLoc != StringRef::npos)
        return prettyName.substr(lastSeparatorLoc);
    }
  }

  return prettyName;
}

static void printSymbol(raw_ostream &os, SymbolRefAttr symbol,
                        SharedState *diagShared) {
  // When mangling, keep things simple.
  if (diagShared == nullptr) {
    std::string nestedSymbolName;
    llvm::raw_string_ostream buff(nestedSymbolName);
    printNestedSymbolReference(buff, symbol);
    os << trimBuiltinNamespace(nestedSymbolName);
    return;
  }

  // When printing for diagnostics and the user, we can cut things down to make
  // them more readable.
  StringRef name = getNameFromSymbolRef(symbol);

  // Remove std:: prefixes.
  os << trimBuiltinNamespace(name);
}

/// Given a parameter list for a function or struct, print it out in a nice
/// user-readable format (e.g. eliding infer-only and defaulted parameters).
///
/// This needs to handle the case when 'paramInfo' is null, e.g. when mangling.
///
/// The 'typesImplied' boolean indicates when we're in a struct - we can omit
/// implicit conversions to tidy up the printout because struct's can't be
/// overloaded on parameter sets like functions are.
static void printParamList(raw_ostream &os, PogListAttr paramInfo,
                           ArrayRef<TypedAttr> params,
                           ASTTypePrinterContext ctx, bool typesImplied) {
  SharedState *diagShared = ctx.shared;
  if (params.empty())
    return;

  SmallVector<std::tuple<StringAttr, TypedAttr, VariadicKind>> paramsToPrint;

  // If we're printing for diagnostics, we'll have 'paramInfo'.  In that case we
  // want to avoid printing defaulted parameter values that are the same as
  // their default value.
  if (paramInfo) {
    assert(paramInfo.size() == params.size() &&
           "Unexpected number of bound params");

    ParameterEvaluator evaluator(params);

    // Find out about default parameter values.
    bool skippedPositional = false;
    for (auto [idx, pog, paramValue] :
         llvm::enumerate(paramInfo.getPogs(), params)) {

      // Inline variadic parameter values if they are known.
      if (pog.isPosVarArg()) {
        auto info = ASTType(paramValue.getType()).getParameterListInfo();
        if (auto values = dyn_cast<ParamListAttr>(info.valueList)) {
          // Each value is passed as an individual value to the vararg, not an
          // unpack.
          for (auto elt : values.getValues())
            paramsToPrint.push_back({StringAttr(), elt, VariadicKind::None});
          continue;
        }
        if (auto structAttr = sugarDynCast<LITStructAttr>(paramValue);
            structAttr && structAttr.getValues().empty()) {
          paramsToPrint.push_back(
              {StringAttr(), info.valueList, VariadicKind::PosVarArg});
          continue;
        }
        // TODO: Should add "*" ahead of splatted lists.
      }

      auto passingKind = pog.getPassingKind();

      // See if this parameter has a default value.  If so, and if the
      // provided value matches it, then don't print the parameter in the
      // list.
      if (auto def = paramInfo.getDefault(idx)) {
        // Make sure to substitute other parameter values in, e.g. so we can
        // handle things like:
        //   struct UnsafePointer[type: AnyType,
        //                        align: Int = _default_alignment[type]()]:
        def = evaluator.getReboundAttribute(def);
        if (isEqualCanon(paramValue, def) &&
            passingKind != PassingKind::PosOnly) {
          // If we skip a posOrKw then include keyword names for any other
          // posOrKw's that come after it.
          skippedPositional |= (passingKind == PassingKind::PosOrKw);
          continue;
        }
      }

      StringAttr name;
      switch (passingKind) {
      case PassingKind::Implicit:
      case PassingKind::Inferred:
        continue; // Don't print implicit parameters at all.
      case PassingKind::PosOnly:
        break; // Never include a name.
      case PassingKind::PosOrKw:
        if (!skippedPositional)
          break; // Don't include a name unless we skipped another one.
        [[fallthrough]];
      case PassingKind::KwOnly:
        name = paramInfo.getName(idx);
        break;
      }
      paramsToPrint.push_back({name, paramValue, VariadicKind::None});
    }

  } else {
    // When generating mangled names, don't include names for parameters since
    // positional information is enough.
    for (TypedAttr paramValue : params)
      paramsToPrint.push_back({StringAttr(), paramValue, VariadicKind::None});
  }

  if (!paramsToPrint.empty()) {
    os << '[';
    llvm::interleaveComma(
        paramsToPrint, os,
        [&](std::tuple<StringAttr, TypedAttr, VariadicKind> param) {
          if (StringAttr name = std::get<0>(param))
            os << name.strref() << '=';

          if (std::get<2>(param) == VariadicKind::PosVarArg)
            os << '*';

          TypedAttr value = std::get<1>(param);
          ASTType::printParam(os, value, ctx,
                              /*hasContextualType=*/typesImplied && diagShared);
        });
    os << ']';
  }
}

/// Print the input parameter types of a generator type/attr.
/// This needs to handle the case when 'paramInfo' is null.
/// An additional attr/type body can be provided that will also be rebound with
/// parameter names (if available in paramInfo) and returned. The provided
/// `evaluator` is populated with the parameter index bindings so callers can
/// reuse them (e.g. `printGeneratorBodyConstraints`) and substitute identically
/// to the parameters and body.
template <typename BodyT>
static BodyT
printGeneratorInterface(raw_ostream &os, ArrayRef<Type> inputParamTypes,
                        PogListAttr paramInfo, ASTTypePrinterContext ctx,
                        BodyT body, ParameterEvaluator &evaluator) {
  assert(paramInfo && "always present");
  SharedState *diagShared = ctx.shared;
  if (!diagShared) {
    os << '[';
    // If no param metadata, just print the types.
    auto printFn = [&](Type type) { ASTType(type).print(os, ctx); };
    llvm::interleaveComma(inputParamTypes, os, printFn);
    os << ']';
    // Since we aren't printing param names, index references must not be
    // remapped to names either.
    for (auto _ : inputParamTypes)
      evaluator.appendIndexBinding(TypedAttr());
    return body;
  }

  bool needComma = false;
  PassingKind lastPassingKind = PassingKind::Implicit;
  for (auto [i, type] : llvm::enumerate(inputParamTypes)) {
    ASTType reboundType = evaluator.getReboundType(type);

    auto curPassingKind = paramInfo.getPassingKind(i);

    // Don't print this if it is an autoparam.
    StringRef name = paramInfo.getName(i).strref();
    if (isHiddenGeneratorParam(curPassingKind, name)) {
      // If this is an origin parameter Foo._mlir_origin, print it as "Foo".
      // These come up in ref [x] specifications all the time.
      if (sugarIsa<OriginType>(reboundType) && name.contains("._mlir_origin")) {
        name = name.substr(0, name.find("._mlir_origin"));
        evaluator.appendIndexBinding(ParamDeclRefAttr::get(name, reboundType));
        continue;
      }
      // Any direct use of this should print as _ since we're not going to show
      // the declaration.
      // Use UnknownAttr instead of UnboundAttr here since UnboundAttr are
      // treated specially in a lot of places. E.g., by `BindParamAttr`. In this
      // case, we mean a concrete value that we don't care (instead of a
      // unprovided parameter values).
      evaluator.appendIndexBinding(UnknownAttr::get(reboundType));
      continue;
    }

    if (!name.empty())
      evaluator.appendIndexBinding(ParamDeclRefAttr::get(name, reboundType));
    else
      evaluator.appendIndexBinding(ParamIndexRefAttr::get(i, reboundType));

    // Handle the param separator.
    if (needComma)
      os << ", ";
    else
      os << "[";
    needComma = true;

    // Print parameter group separators if needed.
    if (lastPassingKind != PassingKind::Implicit) { // Not on first param.
      if (lastPassingKind == PassingKind::PosOnly &&
          curPassingKind != PassingKind::PosOnly) {
        os << "/, ";
        lastPassingKind = PassingKind::PosOrKw;
      }
      if (lastPassingKind == PassingKind::Inferred &&
          curPassingKind != PassingKind::Inferred) {
        os << "//, ";
        lastPassingKind = PassingKind::PosOrKw;
      }
    }

    // Handle variadics correctly.
    if (paramInfo.isPosVarArg(i)) {
      os << "*";
      reboundType = reboundType.getParameterListInfo().elementType;
    } else {
      if (curPassingKind == PassingKind::KwOnly &&
          (i == 0 || lastPassingKind != PassingKind::KwOnly))
        os << "*, ";
    }

    if (!name.empty())
      os << name << ": ";
    reboundType.print(os, diagShared);

    if (TypedAttr defaultOr = paramInfo.getDefault(i)) {
      os << " = ";
      ASTType::printParam(os, defaultOr, diagShared);
    }
    lastPassingKind = curPassingKind;
  };

  // If all the parameters are positional, print the trailing /.
  if (lastPassingKind == PassingKind::PosOnly)
    os << ", /";
  else if (lastPassingKind == PassingKind::Inferred)
    os << ", //";

  if (needComma)
    os << ']';

  // Replace all the *(0, 0) style parameters with names so they print nicely
  // by the client.
  if constexpr (std::is_base_of_v<Attribute, BodyT>)
    body = cast<BodyT>(evaluator.getReboundAttribute(body));
  else
    body = cast<BodyT>(evaluator.getReboundType(body));

  return body;
}

/// Print the trailing `where` clause for a generator's body constraints. The
/// `evaluator` must already carry the parameter index bindings established by
/// `printGeneratorInterface`, so constraint propositions substitute identically
/// to the generator's parameters and body (handling variadic and
/// auto-parameterized params consistently).
static void printGeneratorBodyConstraints(raw_ostream &os,
                                          PogListAttr paramInfo,
                                          ParameterEvaluator &evaluator,
                                          ASTTypePrinterContext ctx) {
  if (!paramInfo || paramInfo.getBodyConstraints().empty())
    return;

  os << " where ";
  llvm::interleave(
      paramInfo.getBodyConstraints(), os,
      [&](ConstraintAttr constraint) {
        TypedAttr prop =
            evaluator.getReboundAttribute(constraint.getProposition());
        ASTType::printParam(os, prop, ctx);
      },
      ", ");
}

/// If the parameter being referenced is an auto-parameterization of the
/// current function or struct, dig it out so we can print the correct name.
/// Consider something like:
///    struct S[a: Scalar]:
///       def f(b: Scalar):
///          use(a.dtype, b.dtype)
/// Both "a.dtype" and "b.dtype" will resolve to a (mangled) string of
/// "dtype", but we would really like to print them as "a.dtype" so the user
/// knows what is going on, and we don't get a T != T error.
///
/// This returns success when handled, failure otherwise.
static void prettyPrintParamName(ParamDeclRefAttr declRef, SharedState &shared,
                                 raw_ostream &os) {
  // If this is an implicit parameter injected due to auto-parameterization,
  // then it will have a uniquing identifier on it, rip that off.
  auto demangledName =
      demangleParameterName(declRef.getName(), /*forUser*/ true);

  // If the name wasn't mangled, then it is a normal user parameter, just
  // print it.
  if (demangledName == declRef.getName()) {
    os << demangledName;
    return;
  }

  ASTDecl *ctxDecl = shared.declResolver->getDiagnosticDeclContext();
  if (!ctxDecl) {
    os << demangledName;
    return;
  }

  // Walk up the decl hierarchy to find the one that contains the parameter.
  auto [curDecl, paramDecls, paramIdx] = ctxDecl->lookupParamReference(declRef);
  if (!curDecl) {
    os << demangledName;
    return;
  }

  // Otherwise, check to see if it is an autoparam.  It could be an autoparam of
  // another parameter, or could be an autoparam for an argument type of a
  // function. Check parameters first (handling structs and functions).
  for (auto paramDecl : paramDecls) {
    for (auto p : ASTType(paramDecl.getType()).getParamBindings()) {
      if (p == declRef) {
        // The param found may itself be an autoparam.  Recurse to print it.
        ASTType::printParam(os, ParamDeclRefAttr::get(paramDecl), {&shared});
        os << "." << demangledName;
        return;
      }
    }
  }

  // If this is a function, it may be an auto-param for an argument type.
  if (auto fnDecl = dyn_cast_if_present<LIT::FnOp>(curDecl->getIfOperation())) {
    auto fnSig = fnDecl.getFuncTypeGenerator();
    for (auto [idx, argType] :
         llvm::enumerate(fnDecl.getFunctionType().getInputs())) {
      auto printArgName = [&]() {
        auto argName = fnSig.getArgName(idx).strref();
        if (!argName.empty())
          os << argName;
        else
          os << "arg" << idx;
      };

      auto userArgType =
          RefType::stripRefConvention(argType, fnSig.getArgConvention(idx));
      if (llvm::is_contained(ASTType(userArgType).getParamBindings(),
                             declRef)) {
        printArgName();
        os << "." << demangledName;
        return;
      }

      // If is possible that this parameter is an autoparam origin or mut bool
      // for a ref argument.  Check to see if that is the case.
      if (auto refType = dyn_cast<RefType>(argType))
        if (auto refOrigin = dyn_cast<ParamDeclRefAttr>(refType.getOrigin())) {
          if (refOrigin.getName() == declRef.getName()) {
            printArgName();
            return;
          }
        }
    }
  }

  os << demangledName;
}

/// Attempt to pretty print the specified ParamIndexRefAttr, returning failure
/// if it didn't work out.
static ParamDeclRefAttr findDeclRefForIndexRef(ParamIndexRefAttr idxRef,
                                               SharedState *shared) {
  if (!shared)
    return {};
  ASTDecl *ctxDecl = shared->declResolver->getDiagnosticDeclContext();
  if (!ctxDecl)
    return {};

  // FIXME: The ASTPrinter needs a notion of current de Bruijn depth. For now we
  // just allow anything, assuming it lines up with the decl we're working on.
  // Need to check `idxRef.getDepth() == 0` here.

  // Find the named param decl from the param list and print a reference
  // to it instead.
  for (; ctxDecl; ctxDecl = ctxDecl->getParentDecl()) {
    if (auto op = ctxDecl->getIfOperation()) {
      if (auto fn = dyn_cast<LIT::FnOp>(op)) {
        auto paramDecls = fn.collectAllParams(/*implicitOrigins*/ false);
        if (idxRef.getIndex() < paramDecls.size())
          return ParamDeclRefAttr::get(paramDecls[idxRef.getIndex()]);

        if (auto declIntf = dyn_cast<DeclInterface>(op)) {
          ArrayRef<ParamDeclAttr> paramDecls = declIntf.getInputParams();
          if (idxRef.getIndex() < paramDecls.size())
            return ParamDeclRefAttr::get(paramDecls[idxRef.getIndex()]);
        }
      }
    }
  }

  // Couldn't find it.
  return {};
}

/// Find the argument name for the specified implicit origin reference.
static StringAttr
findArgNameForImplicitOriginRef(ImplicitOriginRefAttr originRef,
                                SharedState *diagShared) {
  if (!diagShared)
    return {};
  ASTDecl *ctxDecl = diagShared->declResolver->getDiagnosticDeclContext();
  if (!ctxDecl)
    return {};

  // Find the named param decl from the param list and print a reference
  // to it instead.
  for (; ctxDecl; ctxDecl = ctxDecl->getParentDecl()) {
    if (auto op = ctxDecl->getIfOperation()) {
      if (auto fn = dyn_cast<LIT::FnOp>(op)) {
        for (auto [idx, argType] :
             llvm::enumerate(fn.getFuncTypeGenerator().getArguments())) {
          if (auto refType = dyn_cast<RefType>(argType))
            if (refType.getOrigin() == originRef)
              return fn.getFuncTypeGenerator().getArgName(idx);
        }
      }
    }
  }

  return {};
}

/// Pretty print a parameter value.
void ASTType::printParam(raw_ostream &os, TypedAttr param,
                         ASTTypePrinterContext ctx, bool hasContextualType) {
  SharedState *diagShared = ctx.shared;
  if (auto cast = dyn_cast<CastFromBuiltinAttr>(param))
    return printParam(os, cast.getArg(), ctx);
  if (auto cast = dyn_cast<CastToBuiltinAttr>(param))
    return printParam(os, cast.getArg(), ctx);

  // It is pretty common for function arguments to use default conversions
  // from the actual value they want, and may not be an
  // always_inline("builtin") constructor, e.g.:
  //   def example(v: Optional[Int64] = None):
  // Without doing anything fancy, we would get something like:
  //   def example(v: Optional[Int64] = Optional[Int64](None)):
  // Which is literally what is happening, but not very pretty.  To clean this
  // up, check to see if call is to an implicit constructor, and if so, elide
  // the call.
  if (hasContextualType) {
    TypedAttr oldParam = param;
    removeImplicitCtorCall(param, ctx.shared);
    if (param != oldParam) // Only allow one implicit conversion.
      hasContextualType = false;
  }

  auto printOperands =
      [&](ArrayRef<TypedAttr> operands, StringRef separator = ", ",
          StringRef lSeparator = "(", StringRef rSeparator = ")") -> void {
    os << lSeparator;
    llvm::interleave(
        operands, os,
        [&](TypedAttr value) {
          // Don't print extracts out of Int.value.
          if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
            value = extract.getStructValue();
          printParam(os, value, ctx);
        },
        separator);
    os << rSeparator;
  };

  if (auto bindParams = dyn_cast<BindParamsAttr>(param)) {
    printParam(os, bindParams.getGenerator(), ctx);
    printOperands(bindParams.getParamValues(), ", ", "[", "]");
    return;
  }

  auto printGeneratorAttr = [&](GeneratorAttr genAttr) {
    PogListAttr paramList = genAttr.getMetadata();
    ParameterEvaluator evaluator;
    TypedAttr reboundBody =
        printGeneratorInterface(os, genAttr.getInputParamTypes(), paramList,
                                ctx, genAttr.getBody(), evaluator);
    os << ' ';
    printParam(os, reboundBody, ctx);
    printGeneratorBodyConstraints(os, paramList, evaluator, ctx);
  };

  if (auto genAttr = dyn_cast<GeneratorAttr>(param)) {
    os << "comptime";
    printGeneratorAttr(genAttr);
    return;
  }

  if (auto genAttr = dyn_cast<SingletonAttr>(param);
      genAttr && isa<FuncLiteralType>(genAttr.getType())) {
    os << "fn_literal";
    return;
  }

  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param)) {
    printSymbol(os, symbolCst.getSymbol(), diagShared);
    if (!symbolCst.getParamValues().empty())
      printOperands(symbolCst.getParamValues(), ", ", "[", "]");
    return;
  }
  if (auto refPack = dyn_cast<RefPackAttr>(param)) {
    llvm::interleaveComma(refPack.getValues(), os,
                          [&](TypedAttr value) { printParam(os, value, ctx); });
    return;
  }

  // Given the operand passed to a VariadicList/VariadicPack argument, return
  // the operands that are passed through it.
  auto getVariadicOperands = [&](TypedAttr operand) -> SmallVector<TypedAttr> {
    // The call to the ctor is passed by ref to the callee.
    operand = cast<StoreToMemAttr>(operand).getValue();

    // The call to the ctor is a #kgen.param.expr<apply.
    auto apply = cast<ParamOperatorAttr>(operand);
    assert(apply.getOpcode() == POC::Apply && "expected apply of init");
    ArrayRef<TypedAttr> elts;
    if (auto elements = dyn_cast<RefPackAttr>(apply.getOperands().back()))
      elts = elements.getValues();
    else {
      // VariadicList will have a StoreToMem containing an Array, or might have
      // a null pointer.
      if (auto storeToMem =
              dyn_cast<StoreToMemAttr>(apply.getOperands().back()))
        elts = cast<POP::ArrayAttr>(storeToMem.getValue()).getValues();
      else {
        assert(isa<PointerAttr>(apply.getOperands().back()) &&
               "Limited forms for building variadics are supported");
        elts = {};
      }
    }
    SmallVector<TypedAttr> result;
    // Each argument value is passed by reference.
    for (auto element : elts)
      result.push_back(cast<StoreToMemAttr>(element).getValue());
    return result;
  };

  // Print the arguments to an Apply/ApplyResultSlot operator, taking into
  // consideration keyword arguments, variadics etc.
  auto printApplyOperands = [&](ArrayRef<TypedAttr> operands,
                                FnTypeGeneratorType fullSig,
                                bool calleeIsMethod) {
    if (!fullSig)
      return printOperands(operands);

    // Drop the pogs for any result or error slot.
    auto pogs = fullSig.getArgListAttrs().getPogs();
    if (fullSig.hasMemoryOnlyResult())
      pogs = pogs.drop_back();
    if (fullSig.isThrows())
      pogs = pogs.drop_back();

    size_t argOffset = 0;
    if (calleeIsMethod) { // 'self' will already be printed by now.
      argOffset = 1;
      pogs = pogs.drop_front();
    }
    assert(operands.size() == pogs.size() && "Unexpected # of operands");

    os << '(';
    bool needComma = false;
    for (auto [idx, pog, operand] : llvm::enumerate(pogs, operands)) {
      assert(pog.getPassingKind() != PassingKind::Implicit &&
             pog.getPassingKind() != PassingKind::Inferred &&
             "argument lists don't have these passingkinds");
      if (needComma)
        os << ", ";

      // Handle Variadics gracefully.
      switch (pog.getVariadic()) {
      case VariadicKind::None:
      case VariadicKind::KwVarArg: // TODO: KwVarArg is generally underserved.
        break;
      case VariadicKind::PosVarArg:
      case VariadicKind::PackVarArg:
        // These are both StoreToMemAttr of a call to the ctor. Pull out the
        // arguments and print them.
        auto operands = getVariadicOperands(operand);
        printOperands(operands, ", ", "", "");
        needComma |= !operands.empty();
        continue;
      }

      // Include the keyword for keyword-only arguments.
      if (pog.getPassingKind() == PassingKind::KwOnly)
        os << pog.getName().str() << "=";

      // If the arg convention is a memory convention then we'll typically have
      // a StoreToMem to plop it into memory. However, in comptime/type-param
      // contexts the operand may not be wrapped yet — fall back gracefully.
      TypedAttr operandToPrint = operand;
      if (hasAddress(fullSig.getArgConvention(idx + argOffset)))
        if (auto store = sugarDynCast<StoreToMemAttr>(operand))
          operandToPrint = store.getValue();

      printParam(os, operandToPrint, ctx);
      needComma = true;
    }

    os << ')';
  };

  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    ArrayRef<TypedAttr> operands = op.getOperands();

    // Sugar the parameter operators the parser can generate.
    switch (op.getOpcode()) {
    case POC::Apply:
    case POC::ApplyResultSlot: {
      // Check if we're applying a known symbol, in which case we can do some
      // more specialized printing.
      auto [nameAttr, calleeParams] =
          tryGetSymbolNameAndParams(operands.front());
      if (!nameAttr) {
        // If we're calling a parameter of function type, print it as a normal
        // call.
        printParam(os, operands.front(), ctx);
        return printOperands(operands.drop_front());
      }

      ArrayRef<TypedAttr> operandsToPrint = operands.drop_front();
      StringRef name = getNameFromSymbolRef(nameAttr);
      // Don't print conversions of boolean's to i1.
      if (name == "__mlir_i1__" && operands.size() == 2)
        return printParam(os, operands.back(), ctx);

      // Don't print Bool.__init__ wrapping conformance attrs. The Bool wrapper
      // adds no user value and we already print the inner conforms_to in a
      // readable form.
      if (name == "__init__" && operands.size() >= 2 &&
          tryGetTypeNameFromSymbolRef(nameAttr) == "Bool" &&
          isa<TypeConformsToTraitAttr>(operands.back()))
        return printParam(os, operands.back(), ctx);

      // Print arithmetic functions using their mathematical form rather than
      // as dunder method calls.
      static SmallDenseMap<StringRef, StringRef> binaryOpNames{
          {"__add__", " + "},       {"__sub__", " - "},
          {"__mul__", " * "},       {"__mod__", " % "},
          {"__truediv__", " / "},   {"__floordiv__", " // "},
          {"__xor__", " ^ "},       {"__and__", " & "},
          {"__or__", " | "},        {"__lshift__", " << "},
          {"__rshift__", " >> "},   {"__eq__", " == "},
          {"__lt__", " < "},        {"__le__", " <= "},
          {"__in__", " in "},       {"__ne__", " != "},
          {"__gt__", " > "},        {"__ge__", " >= "},
          {"__matmul__", " @ "},    {"__pow__", " ** "},
          {"__is__", " is "},       {"__isnot__", " isnot "},
          {"__contains__", " in "},
      };
      if (auto it = binaryOpNames.find(name); it != binaryOpNames.end()) {
        if (name == "__contains__" && operandsToPrint.size() == 2) {
          TypedAttr swapped[] = {operandsToPrint[1], operandsToPrint[0]};
          return printOperands(swapped, /*separator=*/it->second);
        }

        return printOperands(operandsToPrint, /*separator=*/it->second);
      }

      // Print unary prefix operators using their operator syntax.
      static SmallDenseMap<StringRef, StringRef> unaryOpNames{
          {"__invert__", "not "},
          {"__neg__", "-"},
          {"__pos__", "+"},
      };
      if (auto it = unaryOpNames.find(name); it != unaryOpNames.end()) {
        if (!operandsToPrint.empty()) {
          os << it->second;
          return printParam(os, operandsToPrint.front(), ctx);
        }
        // A unary op with no operands is malformed AST.
        llvm_unreachable("unexpected empty operand list for unary operator");
      }

      // Print `x.__getitem__(args...)` as `x[args...]`
      if (name == "__getitem__" && !operandsToPrint.empty()) {
        printParam(os, operandsToPrint.front(), ctx);
        os << '[';
        llvm::interleaveComma(
            operandsToPrint.slice(1), os,
            [&](const TypedAttr &value) { printParam(os, value, ctx); });
        os << ']';
        return;
      }

      // Try to resolve the symbol to a ASTDecl and then to a FnOp.
      FnOp calleeFn;
      bool calleeIsMethod = false;
      if (diagShared) {
        if (ASTDecl *decl =
                diagShared->getDeclResolver().getDeclForFuncSymbol(nameAttr)) {
          calleeFn = cast<FnOp>(decl->getIfOperation());
          calleeIsMethod = decl->tryGetMethodParentDecl() != nullptr;
        }
      }

      bool calleeIsStatic =
          calleeFn && calleeIsMethod && calleeFn.getIsStatic();

      // If we can tell that this is a method call, print the receiver first.
      if (!operandsToPrint.empty() && calleeIsMethod && !calleeIsStatic) {
        printParam(os, operandsToPrint.front(), ctx);
        os << '.';
        operandsToPrint = operandsToPrint.drop_front();
      } else {
        calleeIsMethod = false;
      }

      // Special case: struct __init__ constructor calls for literal types.
      if (calleeIsStatic && name != "__init__" && diagShared &&
          operands.size() >= 2) {
        // Helper function to check if this is a literal wrapper by name
        auto isLiteralWrapperName = [](StringRef structName) {
          return structName == "StringLiteral" || structName == "IntLiteral" ||
                 structName == "FloatLiteral" || structName == "Origin";
        };

        // Helper function to try printing just the literal value
        auto tryPrintLiteralValue = [&](ArrayRef<TypedAttr> args) -> bool {
          if (args.size() == 1) {
            printParam(os, args[0], ctx);
            return true;
          }
          return false;
        };

        // Primary approach: Use symbol structure to get struct name
        StringRef structName = tryGetTypeNameFromSymbolRef(nameAttr);
        if (isLiteralWrapperName(structName)) {
          if (tryPrintLiteralValue(operandsToPrint))
            return;
        }
      }

      // For constructors, print the type name instead of __init__.
      if (calleeIsStatic && name == "__init__" &&
          nameAttr.getNestedReferences().size() >= 2) {
        // Identity reconstruction: if every callee parameter is an auto-param
        // from the same argument, print just the argument name instead of the
        // full TypeName[arg.field1, arg.field2, ...] expansion.
        if (diagShared && operandsToPrint.empty()) {
          if (StringRef argName =
                  getIdentityReconstructionArgName(calleeParams);
              !argName.empty()) {
            os << argName;
            return;
          }
        }
        os << tryGetTypeNameFromSymbolRef(nameAttr);
      } else {
        // Static methods print 'StructName.method', not just 'method'.
        if (calleeIsStatic)
          os << tryGetTypeNameFromSymbolRef(nameAttr) << '.';

        // Otherwise, print the symbol name.
        printSymbol(os, nameAttr, diagShared);
      }

      // If there are parameters, print them, eliding infer-only and defaulted
      // parameter values.
      FnTypeGeneratorType fullSig;
      if (calleeFn)
        fullSig = calleeFn.getFullSignature();

      // typesImplied=false because we don't want to elide implicit conversions
      // constructor calls, because the function could be overloaded.  We could
      // check to see if the function is not overloaded and elide it.
      printParamList(os, fullSig ? fullSig.getParamListAttrs() : PogListAttr(),
                     calleeParams, ctx,
                     /*typesImplied*/ false);

      // Finally, also print any operands.
      return printApplyOperands(operandsToPrint, fullSig, calleeIsMethod);
    }
    // TODO(MOTO-1632): Operands are never parenthesized, so a nested operand
    // of lower precedence reparses as something else: `(a or b) and c` prints
    // as `a or b and c`, which reads as `a or (b and c)`. Parenthesize an
    // operand whose opcode binds less tightly than its parent's.
    case POC::And:
      llvm::interleave(
          operands, [&](TypedAttr op) { printParam(os, op, ctx); },
          [&] { os << " and "; });
      return;
    case POC::Or:
      llvm::interleave(
          operands, [&](TypedAttr op) { printParam(os, op, ctx); },
          [&] { os << " or "; });
      return;
    case POC::Cond: {
      auto cond = operands[0];
      // Strip _mlir_value extraction for pattern matching.
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(cond))
        cond = extract.getStructValue();

      // Detect and/or patterns from compiler lowering:
      //   A and B  ->  cond(A._mlir_value, B, A)
      //   A or B   ->  cond(A._mlir_value, A, B)
      if (diagShared) {
        if (cond == operands[2]) {
          // "A and B" pattern: condition (stripped) matches else-branch.
          printParam(os, cond, ctx);
          os << " and ";
          printParam(os, operands[1], ctx);
          return;
        }
        if (cond == operands[1]) {
          // "A or B" pattern: condition (stripped) matches then-branch.
          printParam(os, cond, ctx);
          os << " or ";
          printParam(os, operands[2], ctx);
          return;
        }
      }

      // Regular ternary (not a lowered and/or).
      printParam(os, operands[1], ctx);
      os << " if ";
      printParam(os, cond, ctx);
      os << " else ";
      printParam(os, operands[2], ctx);
      return;
    }
    case POC::Rebind:
      // Just omit the types.
      printParam(os, operands.front(), ctx);
      return;
    default:
      // Otherwise, fall back to printing as a parenthesized form like the KGEN
      // printer does.  We don't fall back to the kgen printer because it will
      // print nested subexpressions as KGEN and lose all sugar.
      os << '(' << stringifyEnum(op.getOpcode()) << ' ';
      llvm::interleaveComma(operands, os, [&](TypedAttr operand) {
        printParam(os, operand, ctx);
      });
      os << ')';
      return;
    }
  }
  if (auto getWitness = dyn_cast<GetWitnessAttr>(param)) {
    // Identity reconstruction: simplify Type[arg.f1, arg.f2, ...].witness
    // to arg.witness when the type is rebuilt from one argument's auto-params.
    if (diagShared && tryPrintIdentityReconstruction(
                          os, getWitness.getTypeValue(),
                          getWitness.getWitnessName().strref(), diagShared))
      return;
    printParam(os, getWitness.getTypeValue(), ctx);
    os << "." << getWitness.getWitnessName().strref();
    return;
  }
  if (auto typeAttr = dyn_cast<TypeParamAttr>(param)) {
    ASTType(typeAttr.getMlirType()).print(os, ctx);
    return;
  }
  if (auto upcast = dyn_cast<UpcastAttr>(param))
    return printParam(os, upcast.getInputTypeValue(), ctx);

  if (auto downcast = dyn_cast<DowncastAttr>(param)) {
    printParam(os, downcast.getInputTypeValue(), ctx);
    os << "(";
    ASTType(downcast.getType()).print(os, ctx);
    os << ")";
    return;
  }

  if (auto identical = dyn_cast<ParamIdenticalAttr>(param)) {
    os << "identical(";
    llvm::interleaveComma(identical.getOperands(), os, [&](TypedAttr operand) {
      printParam(os, operand, ctx);
    });
    os << ")";
    return;
  }

  if (auto conformsTo = dyn_cast<TypeConformsToTraitAttr>(param)) {
    if (auto concreteList =
            sugarDynCast<ParamListAttr>(conformsTo.getTypeValue());
        concreteList && concreteList.getValues().empty()) {
      os << "True";
      return;
    }

    // Print as "conforms_to(Type, Trait1 & Trait2)" using valid Mojo syntax,
    // but strip fully-qualified module paths from trait names so we don't leak
    // internal names like "std::builtin::bool::Boolable" into diagnostics and
    // doc output.
    os << "conforms_to(";
    if (auto concreteList =
            sugarDynCast<ParamListAttr>(conformsTo.getTypeValue())) {
      llvm::interleaveComma(concreteList.getValues(), os, [&](TypedAttr value) {
        printParam(os, UpcastAttr::strip(value), ctx);
      });
    } else {
      printParam(os, UpcastAttr::strip(conformsTo.getTypeValue()), ctx);
    }
    os << ", ";
    printParam(os, conformsTo.getTraitType(), ctx);
    os << ")";
    return;
  }

  if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(param)) {
    printParam(os, extractAttr.getStructValue(), ctx);
    // Don't print ._mlir_value in user-facing output; it is an internal
    // implementation detail for accessing the underlying MLIR type.
    if (!diagShared || extractAttr.getField() != "_mlir_value")
      os << '.' << extractAttr.getField().getValue();
    return;
  }

  if (auto variadicCst = dyn_cast<ParamListAttr>(param)) {
    // ParamListAttr appears in a pack list, so it doesn't need extra []'s
    // around it.
    llvm::interleaveComma(variadicCst.getValues(), os,
                          [&](TypedAttr value) { printParam(os, value, ctx); });
    return;
  }

  if (auto reduce = dyn_cast<ParamListReduceAttr>(param)) {
    os << "#" << reduce.name << "(";
    printParam(os, reduce.getParamList(), ctx);
    os << ", base=";
    printParam(os, reduce.getBase(), ctx);
    os << ", reducer=";
    printGeneratorAttr(cast<GeneratorAttr>(reduce.getGenerator()));
    os << ")";
    return;
  }

  if (auto concat = dyn_cast<ParamListConcatAttr>(param)) {
    os << "#" << concat.name << "(";
    printParam(os, concat.getParamLists(), ctx);
    os << ")";
    return;
  }

  if (auto tabulate = dyn_cast<ParamListTabulateAttr>(param)) {
    os << "#" << tabulate.name << "(";
    printParam(os, tabulate.getCount(), ctx);
    os << ", ";
    printGeneratorAttr(cast<GeneratorAttr>(tabulate.getGenerator()));
    os << ")";
    return;
  }

  if (auto size = dyn_cast<ParamListSizeAttr>(param)) {
    os << "len(";
    printParam(os, size.getParamList(), ctx);
    os << ")";
    return;
  }

  if (auto get = dyn_cast<ParamListGetAttr>(param)) {
    printParam(os, get.getParamList(), ctx);
    os << "[";
    printParam(os, get.getIndex(), ctx);
    os << "]";
    return;
  }

  if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(param)) {
    os << dtypeAttr.getDType().getAsString(/*libForm=*/true);
    return;
  }

  // Special case bool constants instead of printing as 0/1.
  if (auto boolAttr = dyn_cast<BoolAttr>(param)) {
    os << (boolAttr.getValue() ? "True" : "False");
    return;
  }
  if (auto simdAttr = dyn_cast<SIMDAttr>(param);
      simdAttr && KGEN::isScalarOf<KGENDType::kBool>(simdAttr.getType())) {
    os << (simdAttr.getAsBool() ? "True" : "False");
    return;
  }

  if (auto noneAttr = dyn_cast<NoneAttr>(param)) {
    os << "None";
    return;
  }

  if (auto strAttr = dyn_cast<StringAttr>(param)) {
    os << '"';
    printAsMojoStringLiteral(strAttr, os);
    os << '"';
    return;
  }

  // Store to mem shows up when comptime values need a reference type. They
  // generally shouldn't be printed (the apply should suck them in) but we
  // handle it in case the printer isn't perfect.
  if (auto storeAttr = dyn_cast<StoreToMemAttr>(param))
    return printParam(os, storeAttr.getValue(), ctx);

  /// A StructAttr is due to an inline @always_inline("builtin") initializer.
  /// Elide it if we have the default type with a literal so we don't print
  /// Int(42), but print it if it is something weird like IntLiteral(42)
  if (auto structAttr = dyn_cast<LITStructAttr>(param)) {
    // Special case for stateless structs.
    if (diagShared && structAttr.getValues().empty()) {
      auto structType = cast<LIT::StructType>(param.getType());
      StringRef typeName = structType.getSymbol().getLeafReference().strref();
      if (typeName == "IntLiteral" || typeName == "FloatLiteral" ||
          typeName == "StringLiteral") {
        assert(structType.getParamValues().size() == 1 &&
               "Literal type should have one parameter");
        printParam(os, structType.getParamValues()[0], ctx);
        return;
      }
      if (typeName == "Origin") {
        assert(structType.getParamValues().size() == 2 &&
               isa<OriginType>(structType.getParamValues()[1].getType()) &&
               "Origin type should have two parameters");
        auto origin = structType.getParamValues()[1];
        printOriginParam(os, origin, diagShared, /*elideOriginOf=*/false);
        return;
      }
    }

    if (auto elt = getSingleElementStructAttr<TypedAttr>(structAttr)) {
      StringRef typeName;
      if (auto structType = sugarDynCast<StructType>(structAttr.getType()))
        typeName = structType.getSymbol().getLeafReference().strref();

      if (typeName == "Bool" || typeName == "SIMDLength") {
        if (auto extract = dyn_cast<LIT::StructExtractAttr>(elt))
          elt = extract.getStructValue();
        printParam(os, elt, ctx);
        return;
      }

      if (typeName == "DType") {
        if (auto dtypeAttr = sugarDynCast<DTypeConstantAttr>(elt)) {
          if (!hasContextualType)
            os << "DType";
          os << '.' << dtypeAttr.getDType().getAsString(/*libForm=*/true);
          return;
        }
      }
      if (typeName == "AddressSpace") {
        if (auto intAttr = getSingleElementStructAttr<IntegerAttr>(elt)) {
          if (intAttr.getValue().isZero()) {
            if (!hasContextualType)
              os << "AddressSpace";
            os << ".GENERIC";
            return;
          }
        }
      }
    }

    // Otherwise do the default "memberwise" printing of a struct, because we
    // don't know where it came from.
    ASTType(structAttr.getType()).print(os, ctx);
    os << '(';
    // TODO: Could print keywords for the labels if there is a reason someday.
    llvm::interleaveComma(structAttr.getValues(), os, [&](auto elt) {
      TypedAttr value = std::get<1>(elt);
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
        value = extract.getStructValue();
      printParam(os, value, ctx);
    });
    os << ')';
    return;
  }

  if (auto convert = dyn_cast<POP::IntLiteralConvertAttr>(param)) {
    printParam(os, convert.getInput(), ctx);
    return;
  }

  if (auto intLitBin = dyn_cast<POP::IntLiteralBinAttr>(param)) {
    const char *binOp = nullptr;
    switch (intLitBin.getOper().getValue()) {
    case POP::IntLiteralBinKind::Add:
      binOp = " + ";
      break;
    case POP::IntLiteralBinKind::Sub:
      binOp = " - ";
      break;
    case POP::IntLiteralBinKind::Mul:
      binOp = " * ";
      break;
    case POP::IntLiteralBinKind::FloorDiv:
      binOp = " // ";
      break;
    case POP::IntLiteralBinKind::Mod:
      binOp = " % ";
      break;
    case POP::IntLiteralBinKind::Lshift:
      binOp = " << ";
      break;
    case POP::IntLiteralBinKind::Rshift:
      binOp = " >> ";
      break;
    case POP::IntLiteralBinKind::And:
      binOp = " & ";
      break;
    case POP::IntLiteralBinKind::Or:
      binOp = " | ";
      break;
    case POP::IntLiteralBinKind::Xor:
      binOp = " ^ ";
      break;
    case POP::IntLiteralBinKind::Pow:
      binOp = " ** ";
      break;
    }

    return printOperands({intLitBin.getLhs(), intLitBin.getRhs()},
                         /*separator=*/binOp);
  }

  if (auto fpLit = dyn_cast<POP::FloatLiteralAttr>(param)) {
    switch (fpLit.getSpecial().getValue()) {
    case POP::FloatLiteralSpecialValues::NegZero:
      os << "-0.0";
      return;
    case POP::FloatLiteralSpecialValues::Inf:
      os << "inf";
      return;
    case POP::FloatLiteralSpecialValues::NegInf:
      os << "-inf";
      return;
    case POP::FloatLiteralSpecialValues::Nan:
      os << "nan";
      return;
    case POP::FloatLiteralSpecialValues::Normal:
      // Convert to f64 to print out the value.
      auto ctx = fpLit.getContext();
      auto f64Type = SIMDType::get(ctx, 1, DType::f64);
      auto simdVal = cast<KGEN::SIMDAttr>(
          POP::FloatLiteralConvertAttr::get(ctx, f64Type, fpLit));
      os << simdVal.getValues()[0].getFloatVal();
      return;
    }
  }

  // Try to resolve an indexRef to a ParamDeclRefAttr for better printing.
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param)) {
    if (auto declRef = findDeclRefForIndexRef(indexRef, diagShared))
      return printParam(os, declRef, ctx);

    os << '$';
    if (size_t depth = indexRef.getDepth())
      os << depth << '|';
    os << indexRef.getIndex();
    return;
  }

  // Origins are handled with their own grammar that has `origin_of(x)` on the
  // outside.
  if (isa<OriginType>(param.getType()))
    return printOriginParam(os, param, diagShared, /*elideOriginOf=*/false);

  // Print ParamDeclRefAttr as the name of the parameter.
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(param)) {
    if (diagShared)
      return prettyPrintParamName(declRef, *diagShared, os);

    // Escape any weird characters in the parameter name that might have
    // been introduced with backticks.
    return printAsMojoStringLiteral(declRef.getName(), os);
  }

  if (auto sugar = dyn_cast<SugarAttr>(param)) {
    // Sugared parameters print as their sugar when always_inline("builtin")
    // even for mangled names because the arguments should be unique.  We don't
    // sugar other things though because the identifiers may not be fully
    // qualified.

    // Identity type reconstruction: simplify MemberAlias sugars like
    // TileTensor[x.dtype, x.layout, ...].rank to just x.rank when the
    // sugared type is rebuilt entirely from one argument's auto-params.
    if (diagShared && sugar.getKind() == SugarKind::MemberAlias &&
        tryPrintIdentityReconstruction(
            os, sugar.getSugared(), sugar.getMemberName().strref(), diagShared))
      return;

    if (diagShared)
      param = sugar.getSugared();
    else
      param = sugar.getExpanded();
    printParam(os, param, ctx);
    if (diagShared && sugar.getKind() == SugarKind::MemberAlias)
      os << '.' << sugar.getMemberName().strref();
    return;
  }

  if (auto eqAttr = dyn_cast<OriginEqAttr>(param)) {
    os << "origin_eq(";
    printOriginParam(os, eqAttr.getLhs(), diagShared, /*elideOriginOf=*/false);
    os << " == ";
    printOriginParam(os, eqAttr.getRhs(), diagShared, /*elideOriginOf=*/false);
    os << ")";
    return;
  }

  if (isa<UnboundAttr>(param)) {
    os << "_";
    return;
  }

  // Handle other KGEN parameters that it knows about with an ugly fallback.
  // TODO: Remove this - we should cover all attrs here, anything that falls
  // back should be an error/assertion.
  os << KGEN::getParamAsString(param);
}

/// Print the specified parameter like we would in a 'ref [x]' argument or
/// result type, e.g. expanding origin sets.
void ASTType::printRefOriginParam(raw_ostream &os, TypedAttr param,
                                  SharedState *diagShared) {
  printOriginParam(os, param, diagShared, /*elideOriginOf=*/true);
}

/// Print the specified parameter like we would in an origin expression, works
/// in an `origin_of(x)` body.
void ASTType::printOriginParam(raw_ostream &os, TypedAttr param,
                               SharedState *diagShared, bool elideOriginOf) {
  struct ASTOriginPrinter : OriginPrinter {
    SharedState *diagShared;

    explicit ASTOriginPrinter(SharedState *diagShared)
        : diagShared(diagShared) {}
    bool isPrettyPrint() const override { return diagShared != nullptr; }
    void printParam(raw_ostream &os, TypedAttr p) const override {
      ASTType::printParam(os, p, {diagShared});
    }
    void printDeclRef(raw_ostream &os,
                      ParamDeclRefAttr declRef) const override {
      if (!diagShared) {
        printAsMojoStringLiteral(declRef.getName(), os);
        return;
      }
      if (ASTDecl *ctxDecl =
              diagShared->declResolver->getDiagnosticDeclContext()) {
        if (sugarIsa<OriginType>(declRef.getType())) {
          IREmitter emitter(*ctxDecl, ExprContext::EC_Origin);
          TypedAttr paramVal =
              emitter.getStdlibOriginOf(declRef, ctxDecl->getLoc());
          if (auto result = dyn_cast<ParamDeclRefAttr>(paramVal))
            declRef = result;
        }
      }

      prettyPrintParamName(declRef, *diagShared, os);
    }

    ParamDeclRefAttr resolveIndexRef(raw_ostream &os,
                                     ParamIndexRefAttr idxRef) const override {
      return findDeclRefForIndexRef(idxRef, diagShared);
    }

    std::optional<llvm::StringRef>
    resolveImplicitOriginRef(raw_ostream &os,
                             ImplicitOriginRefAttr originRef) const override {
      if (StringAttr argName =
              findArgNameForImplicitOriginRef(originRef, diagShared))
        return argName.strref();
      return std::nullopt;
    }

    TypedAttr prepareSugarParam(raw_ostream &os,
                                SugarAttr sugar) const override {
      return diagShared ? sugar.getSugared() : sugar.getExpanded();
    }
  };

  ASTOriginPrinter(diagShared).print(os, param, elideOriginOf);
}

static void printRef(RefType refType, raw_ostream &os,
                     ASTTypePrinterContext ctx) {
  os << "ref";
  bool printed = false;

  // Just print "ref" in the presence of "ref [_]".
  // NOTE: This makes it impossible to know what the mutability is if hard
  // coded - print something like "ref [mut=True]"?
  if (!isa<UnboundAttr>(refType.getOrigin())) {
    os << '[';
    ASTType::printRefOriginParam(os, refType.getOrigin(), ctx.shared);
    printed = true;
  }

  if (!refType.isDefaultAddrSpace()) {
    if (printed)
      os << ", ";
    ASTType::printParam(os, refType.getAddressSpace(), ctx);
    printed = true;
  }
  os << (printed ? "] " : " ");
}

static void printFnGeneratorType(FnOrFnLiteralTypeGeneratorType type,
                                 raw_ostream &os, ASTTypePrinterContext ctx) {
  bool suppressThin = ctx.suppressThin;
  ctx.suppressThin = false;
  if (type.isAsync())
    os << "async ";
  os << "def";
  if (auto fnLiteralGen = type.getIfFnLiteralTypeGenerator()) {
    os << " "
       << getNameFromSymbolRef(
              fnLiteralGen.getSymbolConstantAttr().getSymbol());
  }

  FuncType fnType = type.getBodyFnType();
  auto paramTypes =
      sugarCast<GeneratorType>(type.getAsType()).getInputParamTypes();
  ParameterEvaluator evaluator;
  if (!paramTypes.empty()) {
    fnType = printGeneratorInterface(os, paramTypes, type.getParamListAttrs(),
                                     ctx, fnType, evaluator);
  }
  os << '(';
  bool hadAnyNames = false;
  for (auto [idx, typeX, conventionX] :
       llvm::enumerate(fnType.getArguments(), fnType.getArgConventions())) {
    ASTType type = typeX;
    ArgConvention convention = conventionX;
    if (isResultSlot(convention))
      continue; // Don't print result in argument list.

    if (idx)
      os << ", ";

    // Print / if moving from positional only to PosOrKw or KwOnly.
    auto curPassingKind = fnType.getArgListAttrs().getPassingKind(idx);
    PassingKind lastPassingKind =
        idx ? fnType.getArgListAttrs().getPassingKind(idx - 1) : curPassingKind;
    if (lastPassingKind == PassingKind::PosOnly && idx &&
        (curPassingKind != PassingKind::PosOnly ||
         fnType.getArgListAttrs().getVariadicKind(idx) != VariadicKind::None)) {
      os << "/, ";
      lastPassingKind = PassingKind::PosOrKw;
    }

    // The formal type is VariadicPack[] and the thing to print is a pack
    // attribute, not a type.
    StringAttr name = fnType.getArgName(idx);
    hadAnyNames |= !name.empty();

    const char *stars = "";
    if (fnType.isPosVarArg(idx)) { // Print with the element of the variadic.
      type = RefType::stripRefConvention(type, convention);
      type = ASTType(type).getVariadicListInfo().getElementRefType();
      convention = fnType.getVariadicConvention(idx);
      stars = "*";
    } else if (fnType.isPack(idx)) {
      type = RefType::stripRefConvention(type, convention);
      convention = fnType.getVariadicConvention(idx);
      stars = "*";
    } else if (fnType.isKwVarArg(idx)) {
      type = ASTType(type).getKwargsDictRefValueType();
      stars = "**";
    } else {
      if (curPassingKind == PassingKind::KwOnly &&
          (idx == 0 || lastPassingKind == PassingKind::PosOrKw))
        os << "*, ";
    }

    if (convention == ArgConvention::OwnedMem ||
        convention == ArgConvention::Mut ||
        convention == ArgConvention::ByRefResult ||
        convention == ArgConvention::DeinitMem)
      os << getUserSyntax(convention) << ' ';
    else if (convention == ArgConvention::Ref ||
             convention == ArgConvention::MutRef)
      printRef(cast<RefType>(type), os, ctx);

    os << stars;
    if (!name.empty())
      os << name.getValue() << ": ";

    if (fnType.isPack(idx)) {
      if (name.empty())
        os << ' '; // Don't print "def (**Ts)" when name is missing.
      os << '*';
      TypedAttr variadic = ASTType(fnType.getIfVariadicListOrPack(idx))
                               .getVariadicPackInfo()
                               .typeListStruct;
      ASTType::printParam(os, variadic, ctx);
    } else {
      // For kwargs, already unwrapped `type` to the value type above.
      if (!fnType.isKwVarArg(idx))
        type = RefType::stripRefConvention(type, convention);
      type.print(os, ctx);
    }

    // Check if we are at the end; if so, we might still have to print a
    // '/'. If we're pretty printing for a diagnostic, and don't have any
    // names, then we don't print the trailing slash. This makes the
    // extremely common case of a source signature `def(...) -> ...` look
    // nicer.
    if (hadAnyNames && curPassingKind == PassingKind::PosOnly &&
        idx == fnType.getNumArguments() - 1)
      os << ", /";
  }
  os << ')';
  for (auto [enabled, effect] :
       {std::make_pair(fnType.isThrows(), "raises"),
        std::make_pair(fnType.isCapturing(), "capturing"),
        std::make_pair(fnType.isCABI(), "abi(\"C\")")})

    if (enabled)
      os << ' ' << effect;

  if (fnType.isThrows()) {
    ASTType errorType = fnType.getUserThrownType();
    // Do this the hard way because we may not have diagShared here.
    StringRef typeName;
    if (auto structType = sugarDynCast<LIT::StructType>(errorType))
      typeName = structType.getSymbol().getLeafReference().strref();
    if (typeName != "Error") {
      os << ' ';
      ASTType(errorType).print(os, ctx);
    }
  }

  os << (suppressThin ? " -> " : " thin -> ");
  Type resultType = fnType.getUserResultType();

  if (fnType.isRefResult()) {
    auto refType = cast<RefType>(resultType);
    printRef(refType, os, ctx);
    resultType = refType.getElementType();
  }

  if (isa<KGEN::NoneType>(resultType))
    os << "None";
  else
    ASTType(resultType).print(os, ctx);
  printGeneratorBodyConstraints(os, type.getParamListAttrs(), evaluator, ctx);
}

/// If `decl` is a closure (struct or trait), print its readable source name and
/// return true. A closure may have no source name (it is only set when the
/// signature is available), in which case this returns false and the caller
/// prints the mangled form.
static bool tryPrintClosureSourceName(raw_ostream &os, ASTDecl *decl) {
  if (!decl)
    return false;
  Operation *op = decl->getIfOperation();
  if (auto structOp = dyn_cast_or_null<StructDeclOp>(op)) {
    if (structOp.getDefinesClosure())
      if (auto sourceName = structOp.getSourceName()) {
        os << sourceName->getName().getValue();
        return true;
      }
  } else if (auto traitOp = dyn_cast_or_null<TraitDeclOp>(op)) {
    if (traitOp.getDefinesClosure())
      if (auto sourceName = traitOp.getSourceName()) {
        os << sourceName->getName().getValue();
        return true;
      }
  }
  return false;
}

void ASTType::print(raw_ostream &os, ASTTypePrinterContext ctx) const {
  SharedState *diagShared = ctx.shared;
  if (!mlirType) {
    os << "<<NULL ASTTYPE>>";
    return;
  }

  // Self-substitution: when threaded a selfType, render any sub-expression
  // equal to it as `Self`. This is what makes nested forms like
  // `SIMD[dt, Self.element_size]` print as Self rather than the fully
  // expanded self type.
  if (ctx.selfType && isEqualCanon(ctx.selfType)) {
    os << "Self";
    return;
  }

  Type type = mlirType;
  auto printUserType = [&](SymbolRefAttr symbol, ArrayRef<TypedAttr> params,
                           ASTDecl *typeDecl) {
    // Handle special cases that should be aliased.
    // FIXME(MOCO-367): maintain "typedef" sugar in the type system.
    if (typeDecl &&
        isa_and_nonnull<LIT::StructDeclOp>(typeDecl->getIfOperation())) {
      auto structDecl = cast<LIT::StructDeclOp>(typeDecl->getIfOperation());
      if (params.size() == 2 && structDecl.getDeclName().strref() == "Origin") {
        // Check to see if we have a Bool with a known constant parameter.
        //   #lit.struct<{value: i1 = 1}>
        if (auto value = getSingleElementStructAttr<SIMDAttr>(params[0])) {
          os << (value.getAsBool() ? "MutOrigin" : "ImmOrigin");
          // TODO: While the mlir_origin is almost always infer-only, it can
          // technically be written.  We should print it here if it isn't an
          // infer-only arg value.
          return;
        }
        os << "Origin[mut=";
        printParam(os, params[0], ctx);
        os << "]";
        return;
      }

      // Handle SIMD[dt, 1] with various aliases. Note that the dtype and size
      // may be parametric.
      if (params.size() == 2 && structDecl.getDeclName().strref() == "SIMD") {
        // DType and Int looks like #lit.struct<{value: index = 42}>, dig.
        DTypeConstantAttr dtype =
            getSingleElementStructAttr<DTypeConstantAttr>(params[0]);
        IntegerAttr size = getSingleElementStructAttr<IntegerAttr>(params[1]);
        if (size && size.getInt() == 1) {
          // This list should be kept in sync with the aliases in simd.mojo.
          static std::pair<KGENDType, const char *> dtypeAliases[] = {
              {KGENDType::index, "Int"},
              {KGENDType::uindex, "UInt"},
              {KGENDType::si8, "Int8"},
              {KGENDType::ui8, "UInt8"},
              {KGENDType::si16, "Int16"},
              {KGENDType::ui16, "UInt16"},
              {KGENDType::si32, "Int32"},
              {KGENDType::ui32, "UInt32"},
              {KGENDType::si64, "Int64"},
              {KGENDType::ui64, "UInt64"},
              {KGENDType::si128, "Int128"},
              {KGENDType::ui128, "UInt128"},
              {KGENDType::si256, "Int256"},
              {KGENDType::ui256, "UInt256"},
              {KGENDType::f4e2m1fn, "Float4_e2m1fn"},
              {KGENDType::f6e2m3fn, "Float6_e2m3fn"},
              {KGENDType::f6e3m2fn, "Float6_e3m2fn"},
              {KGENDType::f8e8m0fnu, "Float8_e8m0fnu"},
              {KGENDType::f8e5m2, "Float8_e5m2"},
              {KGENDType::f8e5m2fnuz, "Float8_e5m2fnuz"},
              {KGENDType::f8e4m3fn, "Float8_e4m3fn"},
              {KGENDType::f8e4m3fnuz, "Float8_e4m3fnuz"},
              {KGENDType::f8e3m4, "Float8_e3m4"},
              {KGENDType::bf16, "BFloat16"},
              {KGENDType::f16, "Float16"},
              {KGENDType::f32, "Float32"},
              {KGENDType::f64, "Float64"},
          };
          if (dtype) {
            for (auto [dtypeValue, dtypeName] : dtypeAliases) {
              if (dtype.getDType() == dtypeValue) {
                os << dtypeName;
                return;
              }
            }
          }

          // Otherwise if we know the size is 1, we can use Scalar[] alias,
          // even if the dtype is parametric.
          os << "Scalar[";
          printParam(os, params[0], ctx);
          os << "]";
          return;
        }
      }
    }

    // Only print the leaf reference when pretty printing types.
    printSymbol(os, symbol, diagShared);

    // Print any type parameters if we can find the struct.
    PogListAttr paramInfo;
    if (typeDecl) {
      paramInfo = cast<StructDeclOp>(typeDecl->getIfOperation())
                      .getSignature()
                      .getParamListAttrs();
    }

    printParamList(os, paramInfo, params, ctx,
                   /*typesImplied*/ true);
  };

  if (auto structTy = dyn_cast<StructType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(type).getDecl(*diagShared);
    if (tryPrintClosureSourceName(os, decl))
      return;
    printUserType(structTy.getSymbol(), structTy.getParamValues(), decl);
  } else if (auto anyStruct = dyn_cast<StructMetaType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(anyStruct.getType()).getDecl(*diagShared);
    if (tryPrintClosureSourceName(os, decl))
      return;
    os << "AnyStruct[";
    printUserType(anyStruct.getSymbol(), anyStruct.getParamValues(), decl);
    os << ']';
  } else if (auto traitType = dyn_cast<TraitType>(type)) {
    // Print using the most concise form of the trait composition.
    SmallVector<TraitSymbolAttr> reduced(traitType.getSymbols());
    if (diagShared)
      reduced = reduceTraitCompositionSymbols(*ctx.shared, reduced);

    llvm::interleave(
        reduced, os,
        [&](TraitSymbolAttr traitSymbol) {
          SymbolRefAttr symbol = traitSymbol.getSymbol();
          ASTDecl *decl =
              diagShared
                  ? diagShared->declResolver->getDeclForTypeSymbolIfExists(
                        symbol)
                  : nullptr;
          if (tryPrintClosureSourceName(os, decl))
            return;
          printSymbol(os, symbol, diagShared);
        },
        " & ");
  } else if (auto anyTrait = dyn_cast<AnyTraitType>(type)) {
    os << "AnyTrait[";
    ASTType(anyTrait.getTraitType()).print(os, ctx);
    os << ']';
  } else if (isNoneType()) {
    os << "None";
  } else if (auto ref = dyn_cast<RefType>(type)) {
    printRef(ref, os, ctx);
    ASTType(ref.getElementType()).print(os, ctx);
  } else if (auto variadic = dyn_cast<ParamListType>(type)) {
    os << "KGENParamList[";
    ASTType(variadic.getElementType()).print(os, ctx);
    os << "]";
  } else if (sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(type)) {
    printFnGeneratorType(FnOrFnLiteralTypeGeneratorType::get(type), os, ctx);
  } else if (auto paramRef = dyn_cast<ParamType>(type)) {
    printParam(os, paramRef.getParam(), ctx);
  } else if (auto genType = dyn_cast<GeneratorType>(type)) {
    PogListAttr paramList = genType.getParamListAttrs();
    ParameterEvaluator evaluator;
    os << "__generator_type";
    Type reboundBody =
        printGeneratorInterface(os, genType.getInputParamTypes(), paramList,
                                ctx, genType.getBody(), evaluator);
    os << ' ';
    ASTType(reboundBody).print(os, ctx);
    printGeneratorBodyConstraints(os, paramList, evaluator, ctx);
  } else if (isa<TypeType>(type)) {
    os << "__TypeOfAllTypes";
  } else if (auto originType = dyn_cast<OriginType>(type)) {
    if (originType.isMutableKnown(true))
      os << "LITMutOrigin";
    else if (originType.isMutableKnown(false))
      os << "LITImmOrigin";
    else {
      os << "LITOrigin[";
      printParam(os, originType.isMutable(), ctx);
      os << ']';
    }
  } else if (isa<OriginSetType>(type)) {
    // Use "OriginSet" type name instead of the internal "origin.set"
    os << "LITOriginSet";
  } else if (auto module = dyn_cast<ModuleType>(type)) {
    // Only print the leaf reference when pretty printing types.
    printSymbol(os, module.getSymbol(), diagShared);
  } else if (isa<NeverType>(type)) {
    os << "Never";
  } else {
    // Fall back to printing the raw MLIR type.
    if (!diagShared) {
      os << type;
      return;
    }

    std::string result;
    llvm::raw_string_ostream strstream(result);
    strstream << type;

    // See if we need quote.
    const char *quote = "`";
    if (!result.empty() && std::isalpha(result[0]) &&
        llvm::all_of(result, [](char c) { return std::isalnum(c); }))
      quote = "";
    os << "__mlir_type." << quote << result << quote;
  }
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType astType) {
  if (!astType)
    return os << "<<NULL ASTTYPE>>";
  astType.print(os, ASTTypePrinterContext{});
  return os;
}

std::string ASTType::getAsString(ASTTypePrinterContext ctx) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, ctx);

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(result.begin(), result.end(), '@', '_');
  return os.str();
}

/// Get the specified parameter as a string.
std::string ASTType::getParamAsString(TypedAttr param,
                                      SharedState *diagShared) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printParam(os, param, {diagShared});
  return os.str();
}

/// Get the specified parameter as a string.
std::string ASTType::getOriginAsString(TypedAttr param,
                                       SharedState *diagShared) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printOriginParam(os, param, diagShared, /*elideOriginOf=*/false);
  return os.str();
}
