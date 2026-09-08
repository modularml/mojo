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
// Renders Mojo-syntax signatures directly from MLIR ops. This is the canonical
// implementation used by both the compiler diagnostic path and the mojo-doc
// tool (which delegates here via `PublicASTDecl`). The data model and rendering
// primitives live in `SignatureModel.h`; this file just wires the
// per-op-kind entry points together.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/MojoDiags.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

void M::KGEN::printFunctionSignature(LIT::FnOp fnOp, LIT::SharedState &shared,
                                     llvm::raw_string_ostream &os,
                                     const LIT::ASTDecl *contextDecl,
                                     const SignatureOffsets &offsets) {
  // The DeclResolver context-changer only records the decl; it does not
  // mutate it. Const-cast lets us thread it through from `const`-qualified
  // doc-tooling callers.
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));

  Operation *parentOp = fnOp->getParentOp();
  bool isStatic = fnOp.getIsStatic();
  bool isMethod =
      !isStatic && (isa<StructDeclOp>(parentOp) || isa<TraitDeclOp>(parentOp));
  bool isInit = fnOp.getSpecialFunctionInfo().isInitializer();
  FnTypeGeneratorType signature = fnOp.getFuncTypeGenerator();

  // Self-type substitution for `Self` keyword rendering. `Self` can be uttered
  // by static methods too (e.g. in a return type), so the substitution is
  // gated on the enclosing decl being a struct or trait - not on `isMethod`.
  // The motivation for checking for trait methods comes from printing
  // compiler-synthesized closure `__call__` requirements on closures.`
  std::optional<ASTType> selfType;
  if (auto parentStruct = dyn_cast<StructDeclOp>(parentOp))
    selfType = ASTType(ASTDecl::computeSelfTypeForStruct(parentStruct));
  else if (auto parentTrait = dyn_cast<TraitDeclOp>(parentOp))
    selfType = ASTType(ASTDecl::computeSelfTypeForTrait(parentTrait));

  SmallVector<ParameterInfo, 2> params;
  ParameterEvaluator evaluator =
      populateParameterInfos(shared, signature.getInputParamTypes(),
                             signature.getParamListAttrs(), params, selfType);

  // Function-level constraints (the trailing "where ...").
  std::string fnConstraints;
  if (auto cs = signature.getParamListAttrs().getBodyConstraints(); !cs.empty())
    fnConstraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  SmallVector<ArgumentInfo, 2> args;
  populateArgumentInfos(
      shared, signature, fnOp.getArgumentTypes(), selfType, evaluator,
      [&] { return fnOp.getSpecialFunctionInfo().hasSelfResult(); }, args);

  // Pre-render the return type for the shared printer; suppressed entirely
  // for `__init__`-style functions whose out-arg gets hoisted to the front.
  bool hasOutArgument =
      !args.empty() && args.back().convention == ArgumentConvention::kOut;
  std::string returnTypeStr;
  ASTType resultType = signature.getUserResultType();
  if (!hasOutArgument && resultType && !resultType.isNoneType()) {
    std::optional<ArgConvention> convention;
    if (signature.isRefResult()) {
      convention = ArgConvention::Ref;
      returnTypeStr =
          "ref" + getRefPrefixAsString(shared, cast<RefType>(resultType),
                                       signature, /*isRefResult=*/true);
    }
    Type reboundUserResultType =
        evaluator.getReboundType(fnOp.getUserResultType());
    returnTypeStr +=
        generateTypeString(shared, reboundUserResultType, VariadicKind::None,
                           selfType, convention);
  }

  // Strip the "(...)" mangle suffix from a function source name, leaving just
  // the bare identifier.
  auto stripFunctionMangle = [](StringRef name) {
    return name.split('(').first;
  };

  printFunctionSignatureFromInfos(
      stripFunctionMangle(fnOp.getSourceName().value_or(StringRef())), args,
      params, returnTypeStr, fnConstraints, isInit, isMethod, shared, os,
      offsets);
}

void M::KGEN::printStructSignature(LIT::StructDeclOp structOp,
                                   LIT::SharedState &shared,
                                   llvm::raw_string_ostream &os,
                                   const LIT::ASTDecl *contextDecl,
                                   const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));
  TypeSignatureType signature = structOp.getSignature();
  SmallVector<ParameterInfo, 2> params;
  ParameterEvaluator evaluator =
      populateParameterInfos(shared, signature.getInputParamTypes(),
                             signature.getParamListAttrs(), params);

  // Struct-level constraints (the trailing "where ...").
  std::string constraints;
  if (auto cs = signature.getParamListAttrs().getBodyConstraints(); !cs.empty())
    constraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  printStructSignatureFromInfos(structOp.getName(), params, constraints, shared,
                                os, offsets);
}

void M::KGEN::printAliasSignature(LIT::AliasDeclOp aliasOp,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const LIT::ASTDecl *contextDecl,
                                  const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));
  auto maybeValue = aliasOp.getValue();
  if (!maybeValue)
    return;
  auto generator = dyn_cast<GeneratorAttr>(*maybeValue);
  if (!generator)
    return;
  auto generatorType = dyn_cast<GeneratorType>(generator.getType());
  if (!generatorType)
    return;

  SmallVector<ParameterInfo, 2> params;
  ParameterEvaluator evaluator =
      populateParameterInfos(shared, generatorType.getInputParamTypes(),
                             generatorType.getParamListAttrs(), params);

  // Decl-level constraints (the trailing "where ...").
  std::string constraints;
  if (auto cs = generatorType.getParamListAttrs().getBodyConstraints();
      !cs.empty())
    constraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  auto name = demangleParameterName(aliasOp.getName(), /*forUser=*/true);
  printAliasSignatureFromInfos(name, /*type=*/"", params, constraints, shared,
                               os, offsets);
}

void M::KGEN::printTraitSignature(LIT::TraitDeclOp traitOp,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const LIT::ASTDecl *contextDecl,
                                  const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));

  auto name = demangleParameterName(traitOp.getName(), /*forUser=*/true);
  os << "trait " << name;
}

//===----------------------------------------------------------------------===//
// Diagnostic-oriented helpers
//===----------------------------------------------------------------------===//

std::string M::KGEN::synthesizeDeclSignature(Operation *op, SharedState &shared,
                                             const ASTDecl *contextDecl) {
  if (!op)
    return {};
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::TypeSwitch<Operation *>(op)
      .Case<FnOp>([&](FnOp fnOp) {
        printFunctionSignature(fnOp, shared, os, contextDecl);
      })
      .Case<StructDeclOp>([&](StructDeclOp structOp) {
        printStructSignature(structOp, shared, os, contextDecl);
      })
      .Case<AliasDeclOp>([&](AliasDeclOp aliasOp) {
        printAliasSignature(aliasOp, shared, os, contextDecl);
      })
      .Case<TraitDeclOp>([&](TraitDeclOp traitOp) {
        printTraitSignature(traitOp, shared, os, contextDecl);
      })
      .Default([](Operation *) {});
  return out;
}

bool M::KGEN::hasReadableSourceLocation(Location loc, SharedState &shared) {
  auto &diags = shared.diags;
  auto &sourceMgr = diags.sourceMgr;
  return sourceMgr.FindBufferContainingLoc(diags.convertLocToSMLoc(loc)) != 0;
}

MojoInflightDiag &MojoInflightDiag::attachNote(const ASTDecl &ctxDecl) & {
  auto *shared = getSharedIfActive();
  if (!shared)
    return *this;

  Operation *op = ctxDecl.getIfOperation();
  // Prefer the operation's location if available; it is often more correct in
  // the case of synthetic functions.
  Location loc =
      op ? op->getLoc() : getDiags()->translateLocation(ctxDecl.getLoc());

  auto &note = attachNote(loc);

  // Print synthetic functions differently, mentioning that they're generated
  // functions. If we can, prefer to print with a synthesized decl signature.
  // If we can't, print out the ASTType directly.
  if (auto fnOp = dyn_cast_if_present<FnOp>(op); fnOp && fnOp.isSynthetic()) {
    if (shared) {
      note.addCustomLineText(synthesizeDeclSignature(fnOp, *shared, &ctxDecl));
    } else {
      std::string out;
      llvm::raw_string_ostream os(out);
      os << ASTType(fnOp.getFullSignature());
      note.addCustomLineText(out);
    }
    note.addCustomLineText("    # note - generated function");
    return note;
  }

  // If the location is readable, just defer to adding a regular note
  if (hasReadableSourceLocation(loc, *shared))
    return note;

  // Else synthetize a signature for this op and print that
  if (auto sig = synthesizeDeclSignature(op, *shared, &ctxDecl); !sig.empty()) {
    note.addCustomLineText(sig);
    note.addCustomLineText("    # note - synthetic signature");
  }

  return note;
}

MojoInflightDiag &MojoInflightDiag::attachNote(Location loc, TypedAttr attr) & {
  auto *shared = getSharedIfActive();
  if (!shared)
    return *this;

  auto &note = attachNote(loc);

  if (hasReadableSourceLocation(loc, *shared))
    return note;

  std::string out;
  llvm::raw_string_ostream os(out);
  ASTType::printParam(os, attr, /*ctx=*/{shared});

  if (!out.empty())
    note.addCustomLineText(out);

  return note;
}
