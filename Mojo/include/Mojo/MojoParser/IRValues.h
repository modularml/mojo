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
// This file defines the representations used for expressions emitted to MLIR,
// either as MLIR SSA values for runtime values and addresses, or as MLIR
// attributes for parameter values.
//
// Emitting an expression to MLIR may produce one of the follow representations,
// which can be classified in this type level hierarchy (but notice that PValue
// is both a BValue and RValue):
//
// AnyValue       <- Expr emitted to MLIR...
//   UValue         <- a value with an unresolved type
//     OverloadSetUValue         <- an unresolved overload set
//     InitializerUValue         <- constructor operands {a=1, b="foo"}
//     InferredBaseAttrRefUValue <- inferred-base attr ref `.f64`
//   CValue        <- Concrete value: something with a known type.
//     LValue         <- mutable reference to storage
//       MLValue        <- value is in memory with a mutable reference
//       RLValue        <- with LValue to a 'ref x' binding to initialize.
//       DLValue        <- with dynamic get/set accessors
//     BValue         <- with a borrowed value
//       SBValue        <- value is register-passable and in an SSA register
//       MBValue        <- value is in memory with a reference (may be mutable)
//       MBPValue       <- reference with parametric mutability
//       PMBValue       <- comptime memory borrowed value
//       PValue         <- value is a parameter expression.
//     RValue         <- with an owned value
//       SRValue        <- with a register-passable value in an SSA register
//       MRValue        <- value is in memory with a mutable reference
//       PMRValue       <- comptime memory rvalue
//       PValue         <- with a parameter value
//
// Note that SRValue is not compatible with memory-only types, but MRValue
// can hold any type, including a RegisterPassable type.
//
// One of the important functions of IRValue is to allow the parser to track
// different value kinds without inserting lots of casts everywhere.  In
// particular, MBValue can be a reference with arbitrary mutability because
// IREmitter::emitResult needs to return a borrow to a value emitted to a
// ExprDest, but often the result is ignored.  We wouldn't want the parser to
// insert tons of dead "imm cast" ops into the IR.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_IRVALUES_H
#define KGEN_MOJOPARSER_IRVALUES_H

#include "Mojo/MojoParser/ASTType.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "mlir/IR/Value.h"

namespace M::KGEN::LIT {
class BaseDLValue;
class ExprNode;
class IREmitter;
class OverloadSet;
class FnOp;
class ExprDest;
class StructFieldOp;
class CallOperands;
class AnyValue;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// This enum value keeps track of whether a "target" is being emitted in a
/// var or ref wrapper.  This affects the behavior of a synthesized VarDecl:
enum class PatternDeclKind {
  kNone, // Reuse or synthesize or function-scope variable like Python.
  kVar,  // Make a scoped vardecl in this context. "var x = ..."
  kRef,  // Bind a scoped reference in this context. "ref x = ..."
  kBind, // Bind a scoped variable in this context like an 'imm' arg
         // convention.
};

template <typename ValueType>
struct ASTExprAnd {
  ValueType ir;

  /// This is the expression a value was produced from, carrying location and
  /// additional semantic information.
  const ExprNode *expr;

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTExprAnd()
      : ir(ValueType()), expr(nullptr) {}

  template <typename DerivedValueType>
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG
  ASTExprAnd(DerivedValueType &&ir, const ExprNode *expr)
      : ir(std::move(ir)), expr(expr) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG
  ASTExprAnd(const ASTExprAnd &other) = default;
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTExprAnd &
  operator=(const ASTExprAnd &other) = default;

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG bool isNull() const {
    return ir.isNull();
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG bool operator!() const {
    return !ir;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG explicit
  operator bool() const {
    return bool(ir);
  }

  template <typename OtherValueType>
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG
  operator ASTExprAnd<OtherValueType>() const {
    return {OtherValueType(ir), expr};
  }
};

//===----------------------------------------------------------------------===//
// Value Classifications
//===----------------------------------------------------------------------===//

/// This is used to provide a null representation for the SmartVariant, allowing
/// it to be default constructed to a known state.
class NullRepresentation {};

/// Instances of SRValue model a dynamic value loaded into an SSA value.  This
/// representation can only be used with register-passable types.
class SRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SRValue(Value v) : Value(v) {}

  ASTType getType() const { return ASTType(Value::getType()); }
};

/// Instances of MRValue model a dynamic value stored into memory whose address
/// is represented with an SSA value.  This representation is typically used
/// with memory-only types, but may also be used with register-passable types,
/// (e.g.) when initializing a var declaration.  Values of this type are owned
/// instances of a value that needs to be consumed, akin to an x-value in C++.
class MRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MRValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of MBValue model a borrowed reference to dynamic value stored
/// into memory. The address is represented with an SSA value of !lit.ref type,
/// which might (or might not) be mutable.  See the comment at the top of the
/// file for rationale for why we allow to point to mutable references.
///
/// This representation is used for borrowed arguments, and for some expressions
/// like `a.b` where `a` is an MRValue or MBValue and `b` is a stored property.
class MBValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MBValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of MBPValue model a reference with parametric origin, e.g.
/// derived from a 'ref' argument, or the result of a call that returns a 'ref'.
/// These values are BValue's because they cannot be stored into, but are not
/// MBValue's because the parametric nature of their reference is meaningful
/// (MBValue is never mutable even if the reference it is pointing to says it is
/// fully mutable).
class MBPValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MBPValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of SBValue model a borrowed reference to a dynamic value stored
/// in an SSA register.  This representation is used for borrowed arguments, and
/// for some expressions like `a.b` where `a` is an MRValue or MBValue and `b`
/// is a stored property.
class SBValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SBValue(Value v) : Value(v) {}

  ASTType getType() const { return ASTType(Value::getType()); }
  ASTType getRValueType() const { return getType(); }
};

/// Instances of MLValue model a loadable/storable address as an SSA value with
/// a mutable !lit.ref reference type.
class MLValue : public Value {
public:
  using Value::Value;
  MLValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }
  RefType getRefType() const { return cast<RefType>(getType()); }

private:
  void check() const;
};

/// Instances of RLValue model a storable address as an SSA value with
/// a mutable !lit.ref<!lit.ref<T>> reference type.  This is a reference to a
/// reference. This has a very short half-life: it only gets generated during
/// local `ref` binding, it doesn't live generally in the compiler.
class RLValue : public Value {
public:
  using Value::Value;
  RLValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const {
    return getType().getReferenceElementType().getReferenceElementType();
  }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// DLValue's model a dynamic LValue which has a getter and setter.  Lit
/// supports two ways to spell this - with property access `a.x =`
/// and with subscript syntax `a[i,j] = `, invoking __getattr__/__setattr__ and
/// __getitem__ and __setitem__ respectively.
///
/// DLValues are allowed to be get-only, set-only, or get-set.
class DLValue {
public:
  DLValue() = default;
  DLValue(RCRef<BaseDLValue> storage) : storage(std::move(storage)) {}
  DLValue(const DLValue &existing) : storage(existing.storage.copy()) {}
  DLValue &operator=(const DLValue &existing);
  ~DLValue();

  bool isNull() const { return !storage; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  BaseDLValue *operator->() const { return &*storage; }
  BaseDLValue &operator*() const { return *storage; }

private:
  RCRef<BaseDLValue> storage;
};

/// Instances of PValue model compile time values that are represented as MLIR
/// attributes.  It is both a BValue and an RValue, it may contain both
/// RegisterPassable and memory-only types.
class PValue {
public:
  PValue() = default;
  PValue(TypedAttr v) : storage(v) {}
  PValue(Attribute value) : storage(value) {
    assert((!value || isa<TypedAttr>(value)) && "invalid value attribute");
  }

  PValue(Type value);

  PValue &operator=(TypedAttr newVal) {
    storage = newVal;
    return *this;
  }

  bool isNull() const { return storage == Attribute(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  TypedAttr get() const { return cast_or_null<TypedAttr>(storage); }
  operator TypedAttr() const { return get(); }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const { return get().getType(); }
  ASTType getRValueType() const { return getType(); }

  /// If this value /is/ a type (i.e., if it has metatype type) return it.
  ASTType getIfTypeValue() const;

  const void *getAsOpaquePointer() const {
    return storage.getAsOpaquePointer();
  }

  static PValue getFromOpaquePointer(void *ptr) {
    return {Attribute::getFromOpaquePointer(ptr)};
  }

  void dump() const;

private:
  Attribute storage;
};
raw_ostream &operator<<(raw_ostream &os, PValue value);

/// PMBValue is a PValue that is stored in comptime memory. It's TypeAttr is a
/// StoreToMemAttr.  The type of this attr is a RefType with a comptime origin.
/// As such, it acts like an MBValue, but has a parameter representation.  It is
/// only formed in comptime expressions.
class PMBValue {
public:
  PMBValue() = default;
  PMBValue(TypedAttr v) : storage(v) { check(); }

  PMBValue &operator=(TypedAttr newVal) {
    storage = newVal;
    return *this;
  }

  /// Given an RValue, return a PMBValue that wraps it.
  static PMBValue getFromPValue(PValue value);

  bool isNull() const { return storage == Attribute(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  TypedAttr get() const { return cast_or_null<TypedAttr>(storage); }
  operator TypedAttr() const { return get(); }

  /// Return the RValue that this wraps.
  TypedAttr getUnderlyingRValue() const;

  ASTType getType() const { return get().getType(); }
  RefType getRefType() const { return cast<RefType>(getType()); }
  ASTType getRValueType() const { return getUnderlyingRValue().getType(); }

  void dump() const;

private:
  void check() const;
  TypedAttr storage;
};

/// PMRValue is a PValue that is stored in comptime memory as an owned value.
/// Its TypeAttr is a StoreToMemAttr.  The type of this attr is a RefType with
/// a comptime origin.  As such, it acts like an MRValue, but has a parameter
/// representation.  It is only formed in comptime expressions.
class PMRValue {
public:
  PMRValue() = default;
  PMRValue(TypedAttr v) : storage(v) { check(); }

  PMRValue &operator=(TypedAttr newVal) {
    storage = newVal;
    return *this;
  }

  /// Given an RValue, return a PMRValue that wraps it.
  static PMRValue getFromPValue(PValue value);

  bool isNull() const { return storage == Attribute(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  TypedAttr get() const { return cast_or_null<TypedAttr>(storage); }
  operator TypedAttr() const { return get(); }

  /// Return the RValue that this wraps.
  TypedAttr getUnderlyingRValue() const;

  ASTType getType() const { return get().getType(); }
  RefType getRefType() const { return cast<RefType>(getType()); }
  ASTType getRValueType() const { return getUnderlyingRValue().getType(); }

  void dump() const;

private:
  void check() const;
  TypedAttr storage;
};

/// Instances of OverloadSetUValue represent an unresolved overload set that
/// must be disambiguated before being used.
class OverloadSetUValue {
public:
  OverloadSetUValue(); // Used by dyn_cast.
  OverloadSetUValue(const OverloadSetUValue &existing);
  OverloadSetUValue &operator=(const OverloadSetUValue &existing);
  ~OverloadSetUValue();

  bool isNull() const { return !storage; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  OverloadSet *operator->() { return &**this; }
  const OverloadSet *operator->() const { return &**this; }
  const OverloadSet &operator*() const;
  OverloadSet &operator*();

  template <typename... Args>
  static OverloadSetUValue create(Args &&...args);
  static OverloadSetUValue create(OverloadSet &&set);

private:
  struct OverloadSetWrapper;
  OverloadSetUValue(RCRef<OverloadSetWrapper> storage);

  RCRef<OverloadSetWrapper> storage;
};
raw_ostream &operator<<(raw_ostream &os, OverloadSetUValue value);

/// Instances of InitializerUValue represent the operand list for a constructor
/// call before knowing what the type being constructed is.  It is similar to
/// the {1, 2, 3} syntax in C++: both are resolved when applied to a type, e.g.
/// in the operand list of a function call: array.append({1, 2, 3}).
///
/// This is used in slice operands to subscripts.
class InitializerUValue {
public:
  InitializerUValue(const InitializerUValue &existing);
  InitializerUValue &operator=(const InitializerUValue &existing);
  ~InitializerUValue();

  const enum Syntax {
    kSliceLiteral,   // Operands of 'a[1:2:3]'
    kListLiteral,    // [a, b, c]
    kDictLiteral,    // {a:b, c:d}
    kSetInitLiteral, // {a=42, 17}
  } syntax;

  const CallOperands &get() const;

  static InitializerUValue create(Syntax syntax, CallOperands &&operands);

  /// Given an inferred type for this initializer list, return the operands that
  /// we should use to try to construct it.
  CallOperands getOperandsForInferredType(ASTType type, ExprDest &&dest,
                                          IREmitter &emitter) const;

  /// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
  /// error and return null.
  CValue emitAsCValue(IREmitter &emitter, ExprDest &dest);

  /// If we are binding the literal to a parametric type, we cannot use the
  /// argument type to find the constructor. Instead, default to the type
  /// corresponding to the syntax (list, dict, or set). If the default type does
  /// not conform to the restrictions of the parameter it is a valid failure.
  ASTType getDefaultType(SharedState &shared) const;

private:
  struct ImplWrapper;
  InitializerUValue(Syntax syntax, RCRef<ImplWrapper> storage);
  RCRef<ImplWrapper> storage;
};
raw_ostream &operator<<(raw_ostream &os, InitializerUValue value);

/// Instances of InferredBaseAttrRefUValue represent an attribute reference
/// whose base type is inferred from context, e.g. `.f64` in `foo(.f64)`, or a
/// call of such a reference, e.g. `.hsb_to_rgb(...)` in
/// `foo(.hsb_to_rgb(...))`.
class InferredBaseAttrRefUValue {
public:
  InferredBaseAttrRefUValue() = default; // Used by dyn_cast.
  explicit InferredBaseAttrRefUValue(const ExprNode *expr) : expr(expr) {
    assert(expr && "InferredBaseAttrRefUValue requires an expression");
  }

  bool isNull() const { return !expr; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  const ExprNode *getExpr() const { return expr; }

  /// Emit this as a CValue if it can be resolved, otherwise emit an error and
  /// return null.
  CValue emitAsCValue(IREmitter &emitter, ExprDest &dest);

private:
  const ExprNode *expr = nullptr;
};
raw_ostream &operator<<(raw_ostream &os, InferredBaseAttrRefUValue value);

struct VariantValueStorageBase {
  /// These are all the forms of storage we can have.
  using Storage =
      SmartVariant<NullRepresentation, PValue, SRValue, MRValue,
                   OverloadSetUValue, InitializerUValue,
                   InferredBaseAttrRefUValue, SBValue, MBPValue, PMBValue,
                   PMRValue, MBValue, DLValue, MLValue, RLValue>;

  VariantValueStorageBase()
      : storage(NullRepresentation()) {} // All are default constructible.

  bool isNull() const { return isa<NullRepresentation>(storage); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  Storage &getStorage() { return storage; }
  const Storage &getStorage() const { return storage; }

  // Return true if this is one of the scalar representation.
  bool isSValue() const {
    return isa<SRValue>(storage) || isa<SBValue>(storage);
  }
  // Return true if this is one of the reference representation.
  bool isMValue() const {
    return isa<MBValue, MRValue, MLValue, MBPValue, PMBValue, PMRValue,
               RLValue>(storage);
  }

  /// Given an M*Value, return the underlying reference.
  Value getMValueReference() const;
  RefType getMValueType() const;

  /// Given an S*Value, return the underlying register.
  Value getSValueRegister() const;

  /// Given an S*Value or M*Value, return the underlying register/reference.  If
  /// not, return a null Value.
  Value getMlirValue() const;

  /// If this value is a type, then return it.  This can happen when this is a
  /// PValue with a type metatype (e.g. a computed type) or if it is some other
  /// value that has struct metatype type.
  ASTType getIfTypeValue() const;

protected:
  // This is the actual storage for the representation.
  Storage storage;
};

template <typename DerivedType>
struct VariantValueStorage : public VariantValueStorageBase {
  static DerivedType getFromStorage(const Storage &storage) {
    DerivedType result;
    result.storage = storage;
    return result;
  }
};

template <typename DerivedType>
struct VariantRValue {
  VariantRValue() = default;
  // These are common constructors all RValues have.
  VariantRValue(PValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantRValue(TypedAttr value) : VariantRValue(PValue(value)) {}
  VariantRValue(Attribute value) : VariantRValue(TypedAttr(value)) {
    assert(isa<TypedAttr>(value) && "invalid value attribute");
  }
  VariantRValue(Type value) : VariantRValue(PValue(value)) {}
  VariantRValue(ASTType value) : VariantRValue(PValue(value)) {}
  VariantRValue(SRValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantRValue(MRValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantRValue(PMRValue value) {
    if (value)
      getStorageR() = value;
  }

  PValue getIfPValue() const { return dyn_cast<PValue>(getStorageR()); }
  SRValue getIfSRValue() const { return dyn_cast<SRValue>(getStorageR()); }
  MRValue getIfMRValue() const { return dyn_cast<MRValue>(getStorageR()); }
  PMRValue getIfPMRValue() const { return dyn_cast<PMRValue>(getStorageR()); }

private:
  // These are named getStorageR instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageR() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageR() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// RValue: RValue = PValue|SRValue|MRValue.
class RValue : public VariantValueStorage<RValue>,
               public VariantRValue<RValue> {
public:
  using VariantRValue::VariantRValue;
  using VariantValueStorage::VariantValueStorage;

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<PValue, SRValue, MRValue, PMRValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through a reference to return the element type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, RValue value);

/// This is the base class of any UValue parent, enabling implicit conversion
/// from and checked conversion to child value types.
template <typename DerivedType>
struct VariantUValue {
  VariantUValue() = default;
  // These are common constructors all UValues have.
  VariantUValue(OverloadSetUValue value) {
    if (value)
      getStorageR() = std::move(value);
  }

  VariantUValue(InitializerUValue value) { getStorageR() = std::move(value); }

  VariantUValue(InferredBaseAttrRefUValue value) {
    if (value)
      getStorageR() = std::move(value);
  }

  OverloadSetUValue getIfOverloadSet() const {
    return dyn_cast<OverloadSetUValue>(getStorageR());
  }

  std::optional<InitializerUValue> getIfInitializer() const {
    if (isa<InitializerUValue>(getStorageR()))
      return cast<InitializerUValue>(getStorageR());
    return {};
  }

  InferredBaseAttrRefUValue getIfInferredBaseAttrRef() const {
    return dyn_cast<InferredBaseAttrRefUValue>(getStorageR());
  }

private:
  // These are named getStorageR instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageR() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageR() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// UValue = OverloadSetUValue|InitializerUValue|InferredBaseAttrRefUValue
class UValue : public VariantValueStorage<UValue>,
               public VariantUValue<UValue> {
public:
  using VariantUValue::VariantUValue;
  using VariantValueStorage::VariantValueStorage;

  static UValue getFrom(Storage storage) {
    UValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<OverloadSetUValue, InitializerUValue, InferredBaseAttrRefUValue>(
            storage))
      result.storage = std::move(storage);
    return result;
  }

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, UValue value);

template <typename DerivedType>
struct VariantLValue {
  VariantLValue() = default;
  VariantLValue(MLValue value) {
    if (value)
      getStorageL() = value;
  }
  VariantLValue(RLValue value) {
    if (value)
      getStorageL() = value;
  }
  VariantLValue(DLValue value) { getStorageL() = value; }

  MLValue getIfMLValue() const { return dyn_cast<MLValue>(getStorageL()); }
  RLValue getIfRLValue() const { return dyn_cast<RLValue>(getStorageL()); }
  DLValue getIfDLValue() const { return dyn_cast<DLValue>(getStorageL()); }

private:
  // These are named getStorageL instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageL() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageL() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// LValue = MLValue|RLValue|DLValue.
class LValue : public VariantValueStorage<LValue>,
               public VariantLValue<LValue> {
public:
  using VariantLValue::VariantLValue;
  using VariantValueStorage::VariantValueStorage;

  static LValue getFrom(Storage storage) {
    LValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<MLValue, RLValue, DLValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the reference in a MLValue to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, LValue value);

template <typename DerivedType>
struct VariantBValue {
  VariantBValue() = default;
  VariantBValue(SBValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(MBValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(MBPValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(PValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(PMBValue value) {
    if (value)
      getStorageB() = value;
  }

  SBValue getIfSBValue() const { return dyn_cast<SBValue>(getStorageB()); }
  MBValue getIfMBValue() const { return dyn_cast<MBValue>(getStorageB()); }
  MBPValue getIfMBPValue() const { return dyn_cast<MBPValue>(getStorageB()); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorageB()); }
  PMBValue getIfPMBValue() const { return dyn_cast<PMBValue>(getStorageB()); }

private:
  // These are named getStorageB instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageB() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageB() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// BValue = SBValue|MBValue|MBPValue|PValue|PMBValue.
class BValue : public VariantValueStorage<BValue>,
               public VariantBValue<BValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantValueStorage::VariantValueStorage;

  static BValue getFrom(Storage storage) {
    BValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<SBValue, MBValue, MBPValue, PValue, PMBValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks returns the underlying element type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, LValue value);

/// Concrete Value: CValue = RValue|LValue|BValue.
class CValue : public VariantValueStorage<CValue>,
               public VariantRValue<CValue>,
               public VariantLValue<CValue>,
               public VariantBValue<CValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantLValue::VariantLValue;
  using VariantRValue::VariantRValue;
  using VariantValueStorage::VariantValueStorage;

  CValue() = default;
  CValue(RValue value) { getStorage() = value.getStorage(); }
  CValue(BValue value) { getStorage() = value.getStorage(); }
  CValue(LValue value) { getStorage() = value.getStorage(); }
  CValue(PValue value) {
    if (value)
      storage = value;
  }
  CValue(PMBValue value) {
    if (value)
      storage = value;
  }
  CValue(PMRValue value) {
    if (value)
      storage = value;
  }

  static CValue getFrom(Storage storage) {
    CValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<PValue, SRValue, MRValue, SBValue, MBValue, MBPValue, MLValue,
            RLValue, DLValue, PMBValue, PMRValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Given a value of !lit.ref type, return an MLValue/MBValue/MBPValue
  /// depending on the mutability of the reference.
  static CValue getMValueForRef(Value refValue);

  BValue getIfBValue() const { return BValue::getFrom(getStorage()); }
  RValue getIfRValue() const { return RValue::getFrom(getStorage()); }
  LValue getIfLValue() const { return LValue::getFrom(getStorage()); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorage()); }
  PMBValue getIfPMBValue() const { return dyn_cast<PMBValue>(getStorage()); }
  PMRValue getIfPMRValue() const { return dyn_cast<PMRValue>(getStorage()); }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in memory references to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, CValue value);

/// AnyValue = UValue|CValue (LValue|BValue|RValue).
class AnyValue : public VariantValueStorage<AnyValue>,
                 public VariantRValue<AnyValue>,
                 public VariantLValue<AnyValue>,
                 public VariantBValue<AnyValue>,
                 public VariantUValue<AnyValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantLValue::VariantLValue;
  using VariantRValue::VariantRValue;
  using VariantUValue::VariantUValue;
  using VariantValueStorage::VariantValueStorage;

  AnyValue() = default;

  AnyValue(UValue value) { storage = value.getStorage(); }
  AnyValue(BValue value) { storage = value.getStorage(); }
  AnyValue(RValue value) { storage = value.getStorage(); }
  AnyValue(LValue value) { storage = value.getStorage(); }
  AnyValue(CValue value) { storage = value.getStorage(); }
  AnyValue(PValue value) {
    if (value)
      storage = value;
  }
  AnyValue(PMBValue value) {
    if (value)
      storage = value;
  }
  AnyValue(PMRValue value) {
    if (value)
      storage = value;
  }

  LValue getIfLValue() const { return LValue::getFrom(storage); }
  UValue getIfUValue() const { return UValue::getFrom(storage); }
  CValue getIfCValue() const { return CValue::getFrom(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }
  BValue getIfBValue() const { return BValue::getFrom(storage); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorage()); }
  PMBValue getIfPMBValue() const { return dyn_cast<PMBValue>(getStorage()); }
  PMRValue getIfPMRValue() const { return dyn_cast<PMRValue>(getStorage()); }

  /// Get the RValue type of the value if it can be resolved to one.
  ASTType getRValueTypeIfResolvable() const;

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, AnyValue value);

//===----------------------------------------------------------------------===//
// BaseDLValue classes.
//===----------------------------------------------------------------------===//

/// Subclasses of BaseDLValue model a dynamic LValue which has a computed getter
/// and setter.
class BaseDLValue : public NonAtomicallyReferenceCounted<BaseDLValue> {
public:
  /// This is the RValue type of the value being accessed if known.  It is
  /// inferred from the get/set.
  ASTType elementType;

  BaseDLValue(ASTType elementType) : elementType(elementType) {}

  virtual ~BaseDLValue();
  virtual void print(raw_ostream &os) const = 0;

  virtual CValue emitLoad(ExprDest &dest, IREmitter &emitter) const = 0;
  virtual CValue emitStore(ASTExprAnd<CValue> value,
                           IREmitter &emitter) const = 0;
};

} // namespace M::KGEN::LIT

namespace llvm {

template <typename ActualType>
struct MLIRValueWrapper {
  static const void *getAsVoidPointer(ActualType value) {
    return value.getAsOpaquePointer();
  }
  static ActualType getFromVoidPointer(void *pointer) {
    return ActualType(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MLValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MLValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::RLValue>
    : public MLIRValueWrapper<M::KGEN::LIT::RLValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::SRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::SRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::SBValue>
    : public MLIRValueWrapper<M::KGEN::LIT::SBValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MBValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MBValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MBPValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MBPValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::PValue> {
public:
  using PValue = M::KGEN::LIT::PValue;
  static const void *getAsVoidPointer(PValue value) {
    return value.get().getAsOpaquePointer();
  }
  static PValue getFromVoidPointer(void *pointer) {
    return PValue(cast_or_null<mlir::TypedAttr>(
        mlir::Attribute::getFromOpaquePointer(pointer)));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable
  };
};
} // namespace llvm

#endif // KGEN_MOJOPARSER_IRVALUES_H
