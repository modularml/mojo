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
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/CallOperands.h"
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
  return emitMatchAgainstValue(this, emitter, subject);
}

CValue InferredAttributeRefNode::emitMatch(IREmitter &emitter, CValue subject,
                                           PatternDeclKind patternKind) const {
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

CValue BinOpNode::emitOrMatch(IREmitter &emitter, CValue subject,
                              PatternDeclKind patternKind) const {
  // `pat1 | pat2` matches if either alternative matches. Bindings across
  // alternatives are not supported yet.
  CValue lhsMatch = lhs->emitMatch(emitter, subject, patternKind);
  if (!lhsMatch)
    return {};
  return emitter.emitOrMatchPredicates({lhsMatch, lhs}, [&] {
    return ASTExprAnd<CValue>{rhs->emitMatch(emitter, subject, patternKind),
                              rhs};
  });
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
  // `Type(field=pat, ...)` is a struct pattern: the callee names the expected
  // type, and each operand is a subpattern for a stored field. Positional
  // operands bind in field-declaration order; keywords select by name.
  ASTType patternType = emitter.emitExprType(callee);
  if (!patternType)
    return {};

  ASTType subjectType = subject.getRValueType();
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
