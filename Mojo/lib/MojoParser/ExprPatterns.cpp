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
// This file implements `emitMatch` for expression nodes used as match
// patterns.
//
//===----------------------------------------------------------------------===//

#include "ExprNodes.h"
#include "IREmitter.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/CallOperands.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"

#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// Defined in ExprNodes.cpp; used to project tuple elements for matching.
AnyValue emitGetterSetterAccess(const ExprNode *node, ASTExprAnd<CValue> base,
                                ArrayRef<Operand> exprOperands, ExprDest &dest,
                                IREmitter &emitter);
static CValue emitEnumCaseNameMatch(IREmitter &emitter, CValue subject,
                                    const ExprNode *expr, StringRef caseName,
                                    const ExprNode *typeBase);

//===----------------------------------------------------------------------===//
// Per-ExprNode Support for Matching.
//===----------------------------------------------------------------------===//

CValue ExprNode::emitMatch(IREmitter &emitter, CValue subject,
                           PatternDeclKind patternKind) const {
  emitter.emitError(getLoc(), "expression is not a valid match pattern");
  return {};
}

/// Match a literal / attribute pattern by emitting it as the subject's type
/// and comparing with `__eq__`, then converting to i1.
static CValue emitMatchAgainstValue(const ExprNode *expr, IREmitter &emitter,
                                    CValue subject) {
  // Emit this literal as a value of the subject's type, then compare.
  ExprDest litDest(subject.getRValueType(), EC_MatchSubject);
  AnyValue litValue = emitter.emitExpr(expr, litDest);
  if (!litValue)
    return {};

  CValue eqResult = emitter.emitNamedMethodCall(
      "__eq__",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_BoolCondition),
                   {{AnyValue(subject), expr}, {litValue, expr}}));
  if (!eqResult)
    return {};

  // Convert Bool (or other boolable) to scalar<bool> / i1 for hlcf.elif.
  return emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition);
}

CValue SimpleLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                                    PatternDeclKind patternKind) const {
  // `_` always succeeds and introduces no bindings.
  if (kind == kDiscardLiteral)
    return PValue(SIMDAttr::getScalarBool(emitter.getContext(), true));

  return ExprNode::emitMatch(emitter, subject, patternKind);
}

CValue BoolLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                                  PatternDeclKind patternKind) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue IntLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                                 PatternDeclKind patternKind) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue FloatLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                                   PatternDeclKind patternKind) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue StringLiteralNode::emitMatch(IREmitter &emitter, CValue subject,
                                    PatternDeclKind patternKind) const {
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue DeclRefNode::emitMatch(IREmitter &emitter, CValue subject,
                              PatternDeclKind patternKind) const {
  // Bare identifiers are only valid match patterns when nested under a `var`
  // or `ref` binding. Do not treat them as "match this existing value" — that
  // would create Python's capture-vs-value ambiguity and consume the syntax
  // reserved for a future implicit-binding form (like `for x in ...`).
  // See Mojo/proposals/pattern-matching.md "Future Direction: Implicit
  // Bindings".
  if (patternKind == PatternDeclKind::kNone) {
    emitter.emitError(getLoc(), "bare identifier '")
        << spelling << "' is not a valid match pattern; use 'var " << spelling
        << "' or 'ref " << spelling << "' to bind a name";
    return {};
  }

  // Binding patterns are irrefutable: declare `spelling` under the requested
  // mode and initialize it from the subject. Match subjects are borrowed
  // (BValues), so `var` bindings copy and `ref` bindings borrow — same
  // machinery as `var x = ...` / `ref x = ...` assignment.
  //
  // An elif condition region and its then region are siblings, so a VarDecl
  // emitted into the condition would not dominate uses in the body (or in a
  // guard that shares this scope). Emit the declaration in the parent block
  // before the enclosing elif, then initialize it from the condition region.
  // The first case's predicate is emitted directly into the parent block
  // (it becomes the elif operand), where no hoisting is needed.
  OpBuilder &b = *emitter.builder;
  OpBuilder::InsertPoint condIP = b.saveInsertionPoint();
  if (auto elifOp =
          dyn_cast_if_present<HLCF::ElifOp>(condIP.getBlock()->getParentOp()))
    b.setInsertionPoint(elifOp);

  ExprDest declDest(LValueInitializerType{subject.getRValueType()}, EC_VarInit);
  declDest.setPatternDeclKind(patternKind);
  LValue bindingLV = emitter.emitExprLValue(this, declDest);

  b.restoreInsertionPoint(condIP);
  if (!bindingLV)
    return {};

  ExprDest storeDest(bindingLV, EC_VarInit);
  if (!emitter.emitCResult(subject, this, storeDest))
    return {};

  return PValue(SIMDAttr::getScalarBool(emitter.getContext(), true));
}

CValue AttributeRefNode::emitMatch(IREmitter &emitter, CValue subject,
                                   PatternDeclKind patternKind) const {
  // `Optional.None` / `Type.Case` against an EnumLike subject is a discriminant
  // pattern (no payload). Other attribute refs keep value-equality matching.
  ASTType subjectType = subject.getRValueType();
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumCaseNameMatch(emitter, subject, this, spelling, base);
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue InferredAttributeRefNode::emitMatch(IREmitter &emitter, CValue subject,
                                           PatternDeclKind patternKind) const {
  // `.None` / `.Case` against an EnumLike subject is a discriminant pattern.
  ASTType subjectType = subject.getRValueType();
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumCaseNameMatch(emitter, subject, this, spelling,
                                 /*typeBase=*/nullptr);
  // Resolve `.member` against the subject's type (e.g. `.red` → `Color.red`),
  // then compare for equality like other literal patterns.
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue ParenNode::emitMatch(IREmitter &emitter, CValue subject,
                            PatternDeclKind patternKind) const {
  return subExpr->emitMatch(emitter, subject, patternKind);
}

CValue BinOpNode::emitMatch(IREmitter &emitter, CValue subject,
                            PatternDeclKind patternKind) const {
  if (kind == kOr)
    return emitOrMatch(emitter, subject, patternKind);
  if (kind == kAsPat)
    return emitAsMatch(emitter, subject, patternKind);
  return ExprNode::emitMatch(emitter, subject, patternKind);
}

/// Collect VarDeclOps registered in `scope`, keyed by binding name.
static void
collectPatternBindings(ASTDecl &scope,
                       SmallVectorImpl<std::pair<StringAttr, VarDeclOp>> &out) {
  for (auto &[name, decls] : scope.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto varDecl = dyn_cast_or_null<VarDeclOp>(decl->getIfOperation());
      if (!varDecl)
        continue;
      out.push_back({name, varDecl});
    }
  }
}

/// Verify LHS/RHS or-pattern alternatives bind the same names with matching
/// kinds and types, rewrite RHS stores to use the LHS VarDecls, erase the
/// duplicate RHS VarDeclOps, and promote the LHS bindings into `parentScope`.
static LogicalResult mergeOrPatternBindings(IREmitter &emitter,
                                            ASTDecl &parentScope,
                                            ASTDecl &lhsScope,
                                            ASTDecl *rhsScope, SMLoc loc) {
  SmallVector<std::pair<StringAttr, VarDeclOp>, 4> lhsBindings;
  collectPatternBindings(lhsScope, lhsBindings);

  SmallVector<std::pair<StringAttr, VarDeclOp>, 4> rhsBindings;
  if (rhsScope)
    collectPatternBindings(*rhsScope, rhsBindings);

  // No bindings on either side — nothing to promote.
  if (lhsBindings.empty() && rhsBindings.empty())
    return success();

  // RHS was never emitted (e.g. LHS constant-folded true). Promote LHS only.
  if (!rhsScope) {
    parentScope.mergeDeclsFrom(lhsScope);
    return success();
  }

  if (lhsBindings.empty() || rhsBindings.empty()) {
    StringAttr missing = lhsBindings.empty() ? rhsBindings.front().first
                                             : lhsBindings.front().first;
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << missing.getValue() << "' is bound in one alternative but "
        << "not the other";
    return failure();
  }

  llvm::DenseMap<StringAttr, VarDeclOp> rhsByName;
  for (auto &[name, varDecl] : rhsBindings)
    rhsByName[name] = varDecl;

  for (auto &[name, lhsVar] : lhsBindings) {
    auto it = rhsByName.find(name);
    if (it == rhsByName.end()) {
      emitter.emitError(loc, "or-pattern alternatives must bind the same names")
          << "; '" << name.getValue() << "' is bound in one alternative but "
          << "not the other";
      return failure();
    }
    VarDeclOp rhsVar = it->second;
    if (lhsVar.getKind() != rhsVar.getKind()) {
      emitter.emitError(loc, "or-pattern binding '")
          << name.getValue() << "' must use the same 'var'/'ref' kind in each "
          << "alternative";
      return failure();
    }
    // VarDecl types are `!lit.ref[decl] T`. Each alternative creates its own
    // decl, so the self-origin always differs even when `T` matches. Compare
    // the element types (and address space) instead.
    ASTType lhsType = lhsVar.getType().getElementType();
    ASTType rhsType = rhsVar.getType().getElementType();
    if (!lhsType.isEqualCanon(rhsType)) {
      auto diag = emitter.emitError(loc, "or-pattern binding '")
                  << name.getValue()
                  << "' has incompatible types across alternatives";
      diag.attachNote(loc) << "left alternative has type " << lhsType
                           << ", right has type " << rhsType;
      return failure();
    }

    // Both alternatives write the same name; keep the LHS VarDecl and retarget
    // RHS initializers to it.
    rhsVar.getResult().replaceAllUsesWith(lhsVar.getResult());
    rhsVar->erase();
    rhsByName.erase(it);
  }

  if (!rhsByName.empty()) {
    emitter.emitError(loc, "or-pattern alternatives must bind the same names")
        << "; '" << rhsByName.begin()->first.getValue()
        << "' is bound in one alternative but not the other";
    return failure();
  }

  parentScope.mergeDeclsFrom(lhsScope);
  return success();
}

CValue BinOpNode::emitOrMatch(IREmitter &emitter, CValue subject,
                              PatternDeclKind patternKind) const {
  // `pat1 | pat2` matches if either alternative matches. Bindings introduced
  // by either arm must agree; they are collected in temporary scopes, checked
  // for compatibility, then promoted into the enclosing case scope. This
  // enables matching patterns like "(var x, 4) | (5, var x)", but both sides
  // must bind the same names and to the same types.
  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(this);

  auto createBindingScope = [&](SMLoc scopeLoc) -> ASTDecl & {
    return emitter.getDeclResolver().addFullyResolvedDecl(
        /*declVal=*/nullptr, StringAttr(), scopeLoc, &emitter.declScope);
  };

  // Emit the LHS to catch any nested bindings.
  ASTDecl &lhsScope = createBindingScope(lhs->getLoc());
  IREmitter lhsEmitter(lhsScope, *emitter.builder);
  CValue lhsMatch = lhs->emitMatch(lhsEmitter, subject, patternKind);
  emitter.builder = lhsEmitter.builder;
  if (!lhsMatch)
    return {};

  // Emit the RHS to catch any nested bindings - if the LHS is irrefutable, the
  // RHS doesn't get emitted.
  ASTDecl *rhsScope = nullptr;
  CValue result = emitter.emitOrMatchPredicates({lhsMatch, lhs}, [&] {
    rhsScope = &createBindingScope(rhs->getLoc());
    IREmitter rhsEmitter(*rhsScope, *emitter.builder);
    CValue rhsMatch = rhs->emitMatch(rhsEmitter, subject, patternKind);
    emitter.builder = rhsEmitter.builder;
    return ASTExprAnd<CValue>{rhsMatch, rhs};
  });
  if (!result)
    return {};

  // Verify that they're compatible.
  if (failed(mergeOrPatternBindings(emitter, emitter.declScope, lhsScope,
                                    rhsScope, getLoc())))
    return {};
  return result;
}

CValue BinOpNode::emitAsMatch(IREmitter &emitter, CValue subject,
                              PatternDeclKind patternKind) const {
  // `pattern as name` applies `pattern` and binds `name` to the whole
  // subject without copying. Memory values use `ref`; register-passable
  // (trivial) values have no address, so they use `bind` instead.
  auto *name = dyn_cast<DeclRefNode>(rhs);
  if (!name) {
    emitter.emitError(rhs->getLoc(), "expected a name after 'as'");
    return {};
  }

  // Bind first, while still in the case condition, so the VarDecl is hoisted
  // before the enclosing elif and dominates the case body.
  PatternDeclKind bindKind =
      subject.isMValue() ? PatternDeclKind::kRef : PatternDeclKind::kBind;
  if (!name->emitMatch(emitter, subject, bindKind))
    return {};
  return lhs->emitMatch(emitter, subject, patternKind);
}

CValue UnaryOpNode::emitMatch(IREmitter &emitter, CValue subject,
                              PatternDeclKind patternKind) const {
  // `var`/`ref` patterns are unary wrappers that set the binding mode for
  // their subpattern (e.g. `case var x:` / `case ref (a, b):`).
  if (kind != kVarPat && kind != kRefPat)
    return ExprNode::emitMatch(emitter, subject, patternKind);

  // Nested specifiers like `var ref x` are redundant; keep going with the
  // innermost kind after warning, matching assignment-pattern behavior.
  if (patternKind != PatternDeclKind::kNone &&
      patternKind != PatternDeclKind::kBind) {
    emitter.emitWarning(getLoc()) << "nested 'var' or 'ref' patterns are "
                                     "redundant, remove the outer pattern";
  }

  PatternDeclKind subKind =
      kind == kVarPat ? PatternDeclKind::kVar : PatternDeclKind::kRef;
  return subExpr->emitMatch(emitter, subject, subKind);
}

CValue TupleNode::emitMatch(IREmitter &emitter, CValue subject,
                            PatternDeclKind patternKind) const {
  ASTType subjectType = subject.getRValueType();
  ASTType tupleType = emitter.shared.lookupBuiltinType(
      "Tuple", emitter.getDeclScope(), getLoc());

  if (!tupleType.isEqualCanon(
          subjectType.getWithoutParameters(emitter.shared))) {
    emitter.emitError(getLoc(), "expected a tuple type to match against, got ")
        << subjectType << getRange();
    return {};
  }

  assert(subjectType.getParamBindings().size() == 2 &&
         "Tuple has two parameters");
  auto vaAttr = sugarCast<ParamListAttr>(subjectType.getParamBindings()[0]);
  if (vaAttr.getValues().size() != exprs.size()) {
    emitter.emitError(getLoc(), "cannot match value of ")
        << subjectType << " of " << vaAttr.getValues().size() << " element"
        << plural(vaAttr.getValues().size()) << " against a pattern with "
        << exprs.size() << " element" << plural(exprs.size()) << getRange();
    return {};
  }

  // Empty tuple pattern `()` always matches an empty `Tuple[]`.
  if (exprs.empty())
    return PValue(SIMDAttr::getScalarBool(emitter.getContext(), true));

  // Borrow the subject so each element access can reuse it.
  BValue subjectBVal = emitter.emitBValue({subject, this}, EC_MatchSubject);
  if (!subjectBVal)
    return {};

  // Extract `subject[i]` the same way comptime tuple destructuring does —
  // via a synthesized subscript that prefers `__getitem_param__`.
  auto getTupleItem = [&](ASTType eltType, unsigned index) -> CValue {
    ExprDest eltDest(eltType, EC_TupleElement);
    TypedAttr indexAttr =
        IntegerAttr::get(IndexType::get(emitter.getContext()), index);
    CValue intIndexCValue =
        emitter.emitInt(ASTExprAnd<PValue>{PValue(indexAttr), this},
                        ExprContext::EC_CallParamValue);
    if (!intIndexCValue)
      return {};
    PValue intIndex = intIndexCValue.getIfPValue();
    assert(intIndex && "Int must be PValue when constructed from int attr");

    SyntheticNode indexExpr(getLoc(), intIndex);
    Operand exprOperand(&indexExpr, getLoc(), ArgUnpackStyle::kPositional);
    SubscriptNode subscript(this, this->getLoc(), {}, this->getLoc());
    auto elem = emitGetterSetterAccess(&subscript, {subjectBVal, this},
                                       exprOperand, eltDest, emitter);
    if (!elem) {
      eltDest.resetForError(emitter);
      return {};
    }
    return emitter.emitCValue({elem, this}, EC_TupleElement);
  };

  auto matchElement = [&, patternKind](unsigned index) -> ASTExprAnd<CValue> {
    CValue eltVal = getTupleItem(ASTType(vaAttr.getValues()[index]), index);
    if (!eltVal)
      return {};
    CValue eltMatch = exprs[index]->emitMatch(emitter, eltVal, patternKind);
    return {eltMatch, exprs[index]};
  };

  // Match the first element, then AND each subsequent element with
  // short-circuiting so later patterns (and future bindings) are skipped on
  // failure.
  ASTExprAnd<CValue> combined = matchElement(0);
  if (!combined.ir)
    return {};
  for (unsigned i = 1, e = exprs.size(); i != e; ++i) {
    combined.ir = emitter.emitAndMatchPredicates(
        combined, [&] { return matchElement(i); });
    if (!combined.ir)
      return {};
    combined.expr = exprs[i];
  }
  return combined.ir;
}

CValue CallNode::emitMatch(IREmitter &emitter, CValue subject,
                           PatternDeclKind patternKind) const {
  ASTType subjectType = subject.getRValueType();

  // `Optional.Some(ref elt)` / `Type.Case(...)`: when the subject is
  // `EnumLike`, deep-match the named case and its payload subpatterns instead
  // of treating the call as a struct field pattern.
  if (subjectType.provenConformsToBuiltinTrait("EnumLike", getLoc(),
                                               emitter.shared, {}))
    return emitEnumMatch(emitter, subject, patternKind);

  // `Type(field=pat, ...)` is a struct pattern: the callee names the expected
  // type, and each operand is a subpattern for a stored field. Positional
  // operands bind in field-declaration order; keywords select by name.
  ASTType patternType = emitter.emitExprType(callee);
  if (!patternType)
    return {};

  if (!patternType.isEqualCanon(subjectType)) {
    emitter.emitError(getLoc(), "cannot match value of type ")
        << subjectType << " against pattern type " << patternType << getRange();
    return {};
  }

  auto structType =
      dyn_cast<StructType>(SugarAttr::strip(subjectType.mlirType));
  if (!structType) {
    emitter.emitError(getLoc(), "expected a struct type to match against, got ")
        << subjectType << getRange();
    return {};
  }

  ASTDecl *typeDecl = subjectType.getDecl(emitter.shared);
  if (!typeDecl) {
    emitter.emitError(getLoc(), "cannot match fields of ")
        << subjectType << getRange();
    return {};
  }

  SmallVector<StructFieldOp, 8> storedFields;
  if (auto structDecl =
          dyn_cast_or_null<StructDeclOp>(typeDecl->getIfOperation())) {
    for (auto field : structDecl.getFieldDecls())
      storedFields.push_back(field);
  }

  // Resolve every operand to a field first so unknown/duplicate names are
  // diagnosed even when an earlier subpattern is statically false.
  struct FieldPattern {
    const Operand *operand;
    StructFieldOp fieldOp;
  };
  SmallVector<FieldPattern, 8> fieldPatterns;
  llvm::SmallPtrSet<Attribute, 8> seenFields;
  unsigned nextPositional = 0;

  for (const Operand &operand : operands) {
    if (operand.unpackStyle != ArgUnpackStyle::kPositional &&
        operand.unpackStyle != ArgUnpackStyle::kKeyword) {
      emitter.emitError(operand.getLoc(),
                        "struct patterns do not support unpacked arguments")
          << operand.expr->getRange();
      return {};
    }

    StructFieldOp fieldOp;
    StringAttr fieldName;
    if (operand.isKeyword()) {
      fieldName = operand.name;
      LookupResult lookup = emitter.shared.lookupAndResolveDecl(
          fieldName.getValue(), operand.getLoc(), *typeDecl,
          /*searchParentScopes=*/false);
      if (lookup.isErroneous())
        return {};
      if (!lookup.isSuccess() || lookup.getIfSuccess().size() != 1) {
        emitter.emitError(operand.getLoc(), "'")
            << fieldName.getValue() << "' is not a field of " << subjectType
            << operand.expr->getRange();
        return {};
      }
      fieldOp = dyn_cast_or_null<StructFieldOp>(
          lookup.getIfSuccess().front()->getIfOperation());
      if (!fieldOp) {
        emitter.emitError(operand.getLoc(), "'")
            << fieldName.getValue() << "' is not a stored field of "
            << subjectType << operand.expr->getRange();
        return {};
      }
    } else {
      if (nextPositional >= storedFields.size()) {
        emitter.emitError(operand.getLoc(),
                          "too many positional subpatterns for ")
            << subjectType << " which has " << storedFields.size() << " field"
            << plural(storedFields.size()) << operand.expr->getRange();
        return {};
      }
      fieldOp = storedFields[nextPositional++];
      fieldName = fieldOp.getNameAttr();
    }

    if (!seenFields.insert(fieldName).second) {
      emitter.emitError(operand.getLoc(), "duplicate field '")
          << fieldName.getValue() << "' in struct pattern"
          << operand.expr->getRange();
      return {};
    }
    fieldPatterns.push_back({&operand, fieldOp});
  }

  if (fieldPatterns.empty())
    return PValue(SIMDAttr::getScalarBool(emitter.getContext(), true));

  BValue subjectBVal = emitter.emitBValue({subject, this}, EC_MatchSubject);
  if (!subjectBVal)
    return {};

  auto matchField = [&](const FieldPattern &fp) -> ASTExprAnd<CValue> {
    StructFieldOp fieldOp = fp.fieldOp;
    ASTType fieldType = fieldOp.getReboundType(
        structType, &emitter.shared.getEvaluationContext());
    ExprDest fieldDest(fieldType, EC_AttributeRefBase);
    CValue fieldVal = AttributeRefNode::emitStoredFieldRef(
        {subjectBVal, this}, fieldOp, fp.operand->expr, fieldDest, emitter);
    if (!fieldVal)
      return {};
    CValue fieldMatch =
        fp.operand->expr->emitMatch(emitter, fieldVal, patternKind);
    return {fieldMatch, fp.operand->expr};
  };

  ASTExprAnd<CValue> combined = matchField(fieldPatterns[0]);
  if (!combined.ir)
    return {};
  for (unsigned i = 1, e = fieldPatterns.size(); i != e; ++i) {
    combined.ir = emitter.emitAndMatchPredicates(
        combined, [&] { return matchField(fieldPatterns[i]); });
    if (!combined.ir)
      return {};
    combined.expr = fieldPatterns[i].operand->expr;
  }
  return combined.ir;
}

//===----------------------------------------------------------------------===//
// EnumLike Matching.
//===----------------------------------------------------------------------===//

/// When processing `Type.Case` patterns, require them to be the subject's
/// nominal type.  Allow unbound types like `Optional` to match `Optional[Int]`.
/// Return failure if we emit an error.
static LogicalResult checkEnumCaseTypeBase(IREmitter &emitter, CValue subject,
                                           const ExprNode *typeBase,
                                           const ExprNode *expr) {
  ASTType baseType = emitter.emitExprType(typeBase, /*allowUnbound=*/true);
  if (!baseType)
    return failure();
  ASTType baseNominal = baseType.getWithoutParameters(emitter.shared);
  ASTType subjectNominal =
      subject.getRValueType().getWithoutParameters(emitter.shared);
  if (baseNominal.isEqualCanon(subjectNominal))
    return success();
  emitter.emitError(expr->getLoc(), "cannot match value of type ")
      << subject.getRValueType() << " against enum case of type " << baseType
      << expr->getRange();
  return failure();
}

/// Look up `caseName` in `SubjectType._enum_case_names`. Returns nullopt after
/// emitting an error when the name is not a case.
static std::optional<size_t> lookupEnumCaseIndex(IREmitter &emitter,
                                                 ASTType subjectType,
                                                 StringRef caseName,
                                                 const ExprNode *expr) {
  SyntheticNode typeNode(expr->getLoc(), PValue(subjectType));
  AttributeRefNode namesRef(&typeNode, expr->getLoc(), "_enum_case_names");
  PValue namesPV = emitter.emitExprPValue(&namesRef, EC_AttributeRefBase);
  if (!namesPV)
    return std::nullopt;

  auto namesList = dyn_cast<ParamListAttr>(getCanonicalAttr(namesPV.get()));
  if (!namesList) {
    emitter.emitError(expr->getLoc(), "cannot match on a parametric enum type")
        << expr->getRange();
    return std::nullopt;
  }

  for (auto [idx, nameAttr] : llvm::enumerate(namesList.getValues())) {
    auto nameStr = dyn_cast<StringAttr>(nameAttr);
    if (nameStr && nameStr.getValue() == caseName)
      return idx;
  }
  emitter.emitError(expr->getLoc(), "'")
      << caseName << "' is not a case of " << subjectType << expr->getRange();
  return std::nullopt;
}

/// Return the payload type for case `caseIndex` from `_enum_case_types`. This
/// returns null if an error is emitted.
static ASTType getEnumCasePayloadType(IREmitter &emitter, ASTType subjectType,
                                      unsigned caseIndex,
                                      const ExprNode *expr) {
  SyntheticNode typeNode(expr->getLoc(), PValue(subjectType));
  AttributeRefNode typesRef(&typeNode, expr->getLoc(), "_enum_case_types");
  PValue typesPV = emitter.emitExprPValue(&typesRef, EC_AttributeRefBase);
  if (!typesPV)
    return {};

  auto typesList = dyn_cast<ParamListAttr>(getCanonicalAttr(typesPV.get()));
  if (!typesList || caseIndex >= typesList.getValues().size()) {
    emitter.emitError(expr->getLoc(),
                      "'_enum_case_types' must be a parameter list covering "
                      "every case")
        << expr->getRange();
    return {};
  }

  TypedAttr payloadAttr = typesList.getValues()[caseIndex];
  if (!LIT::isTypeExpr(payloadAttr)) {
    emitter.emitError(expr->getLoc(),
                      "'_enum_case_types' elements must be types")
        << expr->getRange();
    return {};
  }
  return ASTType(payloadAttr);
}

/// True when the case payload is Mojo `NoneType` (no associated value).
static bool isEnumCaseWithoutPayload(IREmitter &emitter, ASTType payloadType,
                                     const ExprNode *expr) {
  if (payloadType.isNoneType())
    return true;
  ASTType noneType = emitter.shared.lookupBuiltinType(
      "NoneType", emitter.getDeclScope(), expr->getLoc());
  if (!noneType)
    return false;
  return payloadType.getWithoutParameters(emitter.shared)
      .isEqualCanon(noneType.getWithoutParameters(emitter.shared));
}

/// Emit `_get_enum_discriminant() == caseIndex` as a scalar bool predicate.
/// This returns the bool result as well as the case number as an Int.
static std::pair<CValue, CValue>
emitEnumDiscriminantMatch(IREmitter &emitter, CValue subject,
                          unsigned caseIndex, const ExprNode *expr) {
  BValue subjectBVal = emitter.emitBValue({subject, expr}, EC_MatchSubject);
  if (!subjectBVal)
    return {{}, {}};

  CValue discriminant = emitter.emitNamedMethodCall(
      "_get_enum_discriminant",
      CallOperands(CallSyntax::kMethodCall, expr, ExprDest(EC_MatchSubject),
                   {{AnyValue(subjectBVal), expr}}));
  if (!discriminant)
    return {{}, {}};

  TypedAttr indexAttr =
      IntegerAttr::get(IndexType::get(emitter.getContext()), caseIndex);
  CValue caseIdxInt = emitter.emitInt(
      ASTExprAnd<PValue>{PValue(indexAttr), expr}, EC_CallParamValue);
  if (!caseIdxInt)
    return {{}, {}};

  CValue eqResult = emitter.emitNamedMethodCall(
      "__eq__",
      CallOperands(
          CallSyntax::kMethodCall, expr, ExprDest(EC_BoolCondition),
          {{AnyValue(discriminant), expr}, {AnyValue(caseIdxInt), expr}}));
  if (!eqResult)
    return {{}, {}};

  return {emitter.emitScalarBool({eqResult, expr}, EC_BoolCondition),
          caseIdxInt};
}

/// Match `Type.Case` / `.Case` (no parentheses) against an EnumLike subject.
/// "expr" may be either an AttributeRefNode or an InferredAttributeRefNode.
/// typeBase is null in the later case.
static CValue emitEnumCaseNameMatch(IREmitter &emitter, CValue subject,
                                    const ExprNode *expr, StringRef caseName,
                                    const ExprNode *typeBase) {
  if (typeBase &&
      failed(checkEnumCaseTypeBase(emitter, subject, typeBase, expr)))
    return {};
  std::optional<size_t> caseIndex =
      lookupEnumCaseIndex(emitter, subject.getRValueType(), caseName, expr);
  if (!caseIndex)
    return {};

  return emitEnumDiscriminantMatch(emitter, subject, *caseIndex, expr).first;
}

//===----------------------------------------------------------------------===//
// EnumLike call patterns (`Type.Case(payload)`).
//===----------------------------------------------------------------------===//

CValue CallNode::emitEnumMatch(IREmitter &emitter, CValue subject,
                               PatternDeclKind patternKind) const {
  // `Optional.Some(ref elt)` / `.Some(pat)`: call form carries payload
  // subpatterns. Cases with no associated value must use `Optional.None` /
  // `.None` without parentheses.
  StringRef caseName;
  if (auto *attr = dyn_cast<AttributeRefNode>(callee)) {
    caseName = attr->spelling;
    if (failed(checkEnumCaseTypeBase(emitter, subject, attr->base, this)))
      return {};
  } else if (auto *inferred = dyn_cast<InferredAttributeRefNode>(callee)) {
    caseName = inferred->spelling;
  } else {
    emitter.emitError(getLoc(),
                      "enum case pattern must be written as 'Type.Case(...)' "
                      "or '.Case(...)'")
        << callee->getRange();
    return {};
  }

  // Figure out what case we're matching against, and the payload type.
  ASTType subjectType = subject.getRValueType();
  auto caseIndex = lookupEnumCaseIndex(emitter, subjectType, caseName, this);
  if (!caseIndex)
    return {};
  FailureOr<ASTType> payloadTypeOrErr =
      getEnumCasePayloadType(emitter, subjectType, *caseIndex, this);
  if (failed(payloadTypeOrErr))
    return {};

  // Reject attempts to pattern match on a None case.
  ASTType payloadType = *payloadTypeOrErr;
  if (isEnumCaseWithoutPayload(emitter, payloadType, this)) {
    emitter.emitError(getLoc(), "enum case '")
        << caseName << "' has no associated value" << getParenRange();
    return {};
  }

  // Empty `Type.Case()` is never valid: no-payload cases omit parentheses,
  // and payload cases need a subpattern.
  if (operands.empty()) {
    emitter.emitError(getLoc(), "enum case '")
        << caseName << "' requires a payload pattern inside the parentheses"
        << getParenRange();
    return {};
  }

  // Reject unsupported unpacking and keyword arguments.
  for (const Operand &operand : operands) {
    if (operand.unpackStyle == ArgUnpackStyle::kKeyword) {
      emitter.emitError(operand.getLoc(),
                        "enum case patterns do not support keyword arguments")
          << operand.expr->getRange();
      return {};
    }
    if (operand.unpackStyle != ArgUnpackStyle::kPositional) {
      emitter.emitError(operand.getLoc(),
                        "enum case patterns do not support unpacked arguments")
          << operand.expr->getRange();
      return {};
    }
  }

  // A single subpattern matches the whole payload.
  // TODO: Support .Case(a, b) as a nested tuple pattern.
  if (operands.size() != 1) {
    emitter.emitError(operands[1].getLoc(),
                      "enum case patterns currently support at most one "
                      "payload subpattern")
        << operands[1].expr->getRange();
    return {};
  }
  // Emit a dynamic check to see if this is the right case.
  auto [discMatch, caseIdxInt] =
      emitEnumDiscriminantMatch(emitter, subject, *caseIndex, this);
  if (!discMatch || !caseIdxInt)
    return {};

  // This lambda generates code to match against the payload of the enum case
  // after the discriminant is matched.
  auto matchPayload = [&]() -> ASTExprAnd<CValue> {
    // Extract the payload reference.
    SyntheticNode subjectNode(getLoc(), subject);
    AttributeRefNode payloadMethod(&subjectNode, getLoc(),
                                   "_unsafe_get_enum_payload");
    SyntheticNode indexNode(getLoc(), caseIdxInt);
    Operand indexOperand(&indexNode, getLoc(), ArgUnpackStyle::kPositional);
    SubscriptNode subscript(&payloadMethod, getLoc(), indexOperand, getLoc());
    CallNode payloadCall(&subscript, getLoc(), /*operands=*/{}, getLoc());
    CValue payload = emitter.emitExprCValue(&payloadCall, EC_MatchSubject);
    if (!payload)
      return {};

    CValue payloadMatch =
        operands[0].expr->emitMatch(emitter, payload, patternKind);
    return {payloadMatch, operands[0].expr};
  };

  // Project the payload and deep-match operands, short-circuiting so
  // `_unsafe_get_enum_payload` runs only when the discriminant matches.
  return emitter.emitAndMatchPredicates({discMatch, this}, matchPayload);
}
