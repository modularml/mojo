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
// This defines the DLValue ("dynamic LValue") implementation details.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DLVALUES_H
#define KGEN_MOJOPARSER_DLVALUES_H

#include "OverloadSet.h"

namespace M::KGEN::LIT {

/// This DLValue implementation represents a discard pattern of _.  It discards
/// its result on store and produces an error if attempting to load it.
class DiscardDLValue : public BaseDLValue {
public:
  const ExprNode *expr;

  DiscardDLValue(ASTType elementType, const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitStore(ASTExprAnd<CValue> value, IREmitter &emitter) const override;
};

/// This DLValue implementation represents a stored attribute projected from
/// another DLValue, e.g. `swap(&a[i].x, ...)`.
class StoredAttributeRefDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  ASTExprAnd<DLValue> baseVal;
  Operation *fieldOp; // StructFieldOp

  StoredAttributeRefDLValue(ASTExprAnd<DLValue> baseVal, StructFieldOp fieldOp,
                            ASTType elementType, const ExprNode *expr);

  StructFieldOp getField() const;

  void print(raw_ostream &os) const override;
  CValue emitLoad(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitStore(ASTExprAnd<CValue> value, IREmitter &emitter) const override;
};

/// This DLValue implementation represents property access `a.x =`
/// and with subscript syntax `a[i,j] = `, invoking __getattr__/__setattr__ and
/// __getitem__ and __setitem__ respectively.
///
/// We allow DLValues to have getter+setter or just setter.
class SubscriptDLValue : public BaseDLValue {
public:
  /// The getter and setter to use; these may both be null.
  PValue getter;
  /// They keyword argument name for the newValue.
  StringAttr setterValueName;

  // Positional operands (including self) for the setter/getter call.
  CallOperands operands;

  /// Return true if this is a subscript, false if this is an attribute access.
  bool isSubscript() const;

  SubscriptDLValue(PValue getter, StringAttr setterValueName,
                   CallOperands &&operands, ASTType elementType);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitStore(ASTExprAnd<CValue> value, IREmitter &emitter) const override;
};

/// This DLValue implementation represents tuple lvalues, e.g. `(a[i], b) = x`.
class TupleDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  // These are the LValues for the sub-elements.
  SmallVector<ASTExprAnd<AnyValue>, 4> eltLValues;

  TupleDLValue(ArrayRef<ASTExprAnd<AnyValue>> eltLValues, ASTType tupleType,
               const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitStore(ASTExprAnd<CValue> value, IREmitter &emitter) const override;
};

/// This DLValue implementation represents list-pattern lvalues, e.g.
/// `[a, b] = x` or `var [a, b] = x`.
class ListPatternDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  // These are the subexpressions of the list pattern; emitted on store.
  SmallVector<ExprNode *, 4> eltExprs;
  // Captured from the assignment dest so `var [a, b] = …` declares elements.
  PatternDeclKind patternDeclKind;

  ListPatternDLValue(ArrayRef<ExprNode *> eltExprs, ASTType listType,
                     PatternDeclKind patternDeclKind, const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitStore(ASTExprAnd<CValue> value, IREmitter &emitter) const override;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DLVALUES_H
