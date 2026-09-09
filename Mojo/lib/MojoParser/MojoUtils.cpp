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
// This file implements common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include "MojoUtils.h"

#include "IREmitter.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ExprNode.h"
#include "Mojo/MojoParser/MojoDiags.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "ParamInf.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

TypedAttr LIT::getOriginsAccessibleByParams(PogListAttr paramList,
                                            ArrayRef<ParamDeclAttr> params,
                                            SharedState &shared,
                                            TypedAttr captureOrigins) {

  // We also need to find all accessible origin sets, even if they are
  // parametric, and union them with the found origins. We don't need to
  // recurse into any nested parameter origins. Even if they contain origin
  // set references, they may not be within the top-level parameter scope, and
  // also we know they can't be accessed by the current function. For example,
  //
  //   def foo[f: def[g: def() capturing [_] -> None] -> None]():
  //       pass
  //
  // `foo` doesn't access the inner origin set of `g` through `f`, because
  // `foo` cannot call `f` without constructing and passing a closure.
  //
  // We can union the sets together by wrapping them in a origin set union.
  // The mutability doesn't matter since it will get flattened.
  SmallVector<TypedAttr> origins;
  auto addOriginSet = [&](TypedAttr param) {
    origins.push_back(OriginSetUnionAttr::get(
        param, OriginType::get(shared.getContext(), /*isMutable=*/true)));
  };

  for (auto [param, pog] : llvm::zip(params, paramList.getPogs())) {
    // Implicit parameters in result slots are not visible on the callee side,
    // so we don't consider their origin accesses.
    if (pog.getPassingKind() == PassingKind::Implicit)
      continue;
    if (sugarIsa<OriginSetType>(param.getType()))
      addOriginSet(ParamDeclRefAttr::get(param));
  }
  if (captureOrigins)
    addOriginSet(captureOrigins);

  return OriginSetAttr::get(shared.getContext(), origins);
}

ASTType LIT::getBoundCoroutineType(ASTDecl &declScope, const ExprNode *expr,
                                   FnTypeGeneratorType sig, TypedAttr origin) {
  auto &shared = declScope.getShared();
  SMLoc loc = expr->getLoc();
  ASTDecl *decl = sig.isThrows() ? shared.getBuiltinRaisingCoroutineType(loc)
                                 : shared.getBuiltinCoroutineType(loc);
  if (!decl) {
    shared.emitError(loc,
                     "internal error: could not find builtin 'Coroutine' type");
    return {};
  }
  ASTType resultType = ASTType(sig.getUserResultType());
  ParamBindings paramBinds(declScope, expr);
  paramBinds.add(expr, PValue(resultType));
  paramBinds.add(expr, origin);

  auto structOp = cast<StructDeclOp>(decl->getIfOperation());
  TypeSignatureType structSig = structOp.getSignature();
  ParamInf inference(paramBinds, structSig.getParamTypes(),
                     structSig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, decl,
                     /*discardError=*/false);
  VerifiedParamBindings bindings = inference.inferForStruct();
  if (!bindings)
    return {};

  return bindings.specializeStructType(structOp);
}

CValue LIT::materializeAsyncCallAsCoroutine(IREmitter &emitter,
                                            AsyncCallOp call,
                                            const ExprNode *expr,
                                            FnTypeGeneratorType sig,
                                            ExprDest &dest) {
  // Collect any origins uttered in any operands or the capture set.
  SmallVector<Type> operandTypes;
  for (Value value : call.getOperands())
    operandTypes.push_back(value.getType());
  SmallVector<TypedAttr> origins =
      emitter.shared.cachedOriginFinder.findOriginsIn(
          operandTypes, call.getCalleeType().getBody().getCaptureOrigins());
  auto argumentsOrigin = OriginSetAttr::get(call.getContext(), origins);

  ASTType coroutineType =
      getBoundCoroutineType(emitter.getDeclScope(), expr, sig, argumentsOrigin);
  if (!coroutineType) {
    dest.resetForError(emitter);
    return {};
  }

  return emitter.emitConstructorCall(
      coroutineType, CallOperands(CallSyntax::kImplicitConvert, expr,
                                  std::move(dest), {{SRValue(call), expr}}));
}

void LIT::markRegionUnreachable(Region *deadRegion, Location unreachableLoc) {
  // Erase bottom up to avoid deleting an op while something uses its results.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(deadRegion->front()))) {
    // Avoid erasing ops that correspond to lazily resolved decls.
    if (isa<ImportOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(op))
      continue;
    op.erase();
  }

  auto builder = OpBuilder::atBlockEnd(&deadRegion->front());
  UnreachableOp::create(builder, unreachableLoc);
}

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

bool LIT::isInternalName(StringRef name) { return name.starts_with('_'); }

namespace {
class IndexRefToNamedRefReplacer
    : public IndexParameterReplacer<IndexRefToNamedRefReplacer> {
public:
  IndexRefToNamedRefReplacer(ArrayRef<ParamDeclAttr> explicitParamDecls,
                             ArrayRef<ParamDeclAttr> implicitOriginDecls)
      : explicitParamDecls(explicitParamDecls),
        implicitOriginDecls(implicitOriginDecls) {}

  Attribute tryReplace(Attribute attr, size_t depth) {
    if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr)) {
      if (indexRef.getDepth() == depth &&
          indexRef.getIndex() < explicitParamDecls.size()) {
        // Replace with the name decl, but reuse the type on the index ref to
        // preserve sugar.
        return ParamDeclRefAttr::get(
            explicitParamDecls[indexRef.getIndex()].getName(),
            replace(indexRef.getType()));
      }
      return {};
    }
    if (auto originRef = dyn_cast<ImplicitOriginRefAttr>(attr)) {
      if (originRef.getDepth() == depth &&
          originRef.getIndex() < implicitOriginDecls.size())
        return ParamDeclRefAttr::get(
            implicitOriginDecls[originRef.getIndex()].getName(),
            originRef.getType());
    }
    return {};
  }

  Type tryReplace(Type, size_t) { return {}; }

private:
  ArrayRef<ParamDeclAttr> explicitParamDecls;
  ArrayRef<ParamDeclAttr> implicitOriginDecls;
};
} // namespace

Type LIT::replaceIndexRefsWithNamedRefs(
    Type type, ArrayRef<ParamDeclAttr> explicitParamDecls,
    ArrayRef<ParamDeclAttr> implicitOriginDecls) {
  if (explicitParamDecls.empty() && implicitOriginDecls.empty())
    return type;
  IndexRefToNamedRefReplacer replacer(explicitParamDecls, implicitOriginDecls);
  return replacer.replace(type);
}

Type LIT::replaceIndexRefsWithNamedRefs(
    Type type, ArrayRef<ParamDeclAttr> explicitParamDecls) {
  return replaceIndexRefsWithNamedRefs(type, explicitParamDecls, {});
}

FunctionType LIT::replaceIndexRefsWithNamedRefs(
    FunctionType functionType, ArrayRef<ParamDeclAttr> explicitParamDecls,
    ArrayRef<ParamDeclAttr> implicitOriginDecls) {
  if (explicitParamDecls.empty() && implicitOriginDecls.empty())
    return functionType;
  IndexRefToNamedRefReplacer replacer(explicitParamDecls, implicitOriginDecls);
  return replacer.replace(functionType);
}

FunctionType
LIT::replaceIndexRefsWithNamedRefs(FunctionType functionType,
                                   ArrayRef<ParamDeclAttr> explicitParamDecls) {
  return replaceIndexRefsWithNamedRefs(functionType, explicitParamDecls, {});
}
