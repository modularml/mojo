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
// This file declares types for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITTYPES_H
#define KGEN_LITDIALECT_LITTYPES_H

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/LITDialect/LITAttrs.h"

namespace M::KGEN {
class ConstraintAttr;
class ParameterExprArrayAttr;
namespace LIT {
class RefPackType;
} // namespace LIT
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/LITDialect/LITTypes.h.inc"

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// FnTypeWrapperGeneratorType
//===----------------------------------------------------------------------===//

// A CRTP base class for FnTypeGeneratorType and FnLiteralTypeGeneratorType that
// wraps a FuncType body and forwards signature queries to it.
template <typename SubClass, typename BaseClass>
class FnTypeWrapperGeneratorType : public BaseClass {
public:
  using BaseT = BaseClass;
  using BaseClass::BaseClass;

  FuncType getBodyFnType() {
    return static_cast<SubClass *>(this)->getBodyFnType();
  }

  //===--------------------------------------------------------------------===//
  // Acting as a GeneratorType
  //===--------------------------------------------------------------------===//

  PogListAttr getParamListAttrs() {
    return static_cast<SubClass *>(this)->getParamListAttrs();
  }
  StringAttr getParamName(size_t idx) {
    return getParamListAttrs().getName(idx);
  }

  //===--------------------------------------------------------------------===//
  // Acting as a FuncType
  //===--------------------------------------------------------------------===//

  FunctionType getValues() { return getBodyFnType().getValues(); }

  llvm::ArrayRef<ArgConvention> getArgConventions() {
    return getBodyFnType().getArgConventions();
  }
  FnEffects getFnEffects() { return getBodyFnType().getFnEffects(); }

  /// Helper to return the argument and result types.
  ArrayRef<Type> getArguments() { return getBodyFnType().getArguments(); }
  Type getArgument(size_t i) { return getArguments()[i]; }
  ArrayRef<Type> getResults() { return getBodyFnType().getResults(); }

  bool hasMemoryOnlyResult() { return getBodyFnType().hasMemoryOnlyResult(); }

  bool isThrows() { return getFnEffects().isThrows(); }
  bool isAsync() { return getFnEffects().isAsync(); }
  bool isCapturing() { return getFnEffects().isCapturing(); }
  bool isRefResult() { return getFnEffects().isRefResult(); }

  /// Return the convention for the specified value argument.
  ArgConvention getArgConvention(size_t inputNo) {
    return getArgConventions()[inputNo];
  }

  size_t getNumArguments() { return getArguments().size(); }
  size_t getNumResults() { return getResults().size(); }

  size_t getNumAsyncReturnSlots() {
    return getBodyFnType().getNumAsyncReturnSlots();
  }

  /// Get the signature metadata.
  FnMetaOriginDataAttr getFnMetaOriginData() {
    return cast<FnMetaOriginDataAttr>(
        getBodyFnType().getMetadataAttr().getMetadata());
  }

  FnMetadataAttr getFnMetadata() {
    return cast<FnMetadataAttr>(getBodyFnType().getMetadataAttr());
  }

  /// Get the argument list metadata.
  PogListAttr getArgListAttrs() { return getBodyFnType().getArgListAttrs(); }

  /// Return the name for the argument at the specified index.
  StringAttr getArgName(size_t idx) { return getArgListAttrs().getName(idx); }

  /// Get the origin set of the capture lifetimes.
  TypedAttr getCaptureOrigins() { return getBodyFnType().getCaptureOrigins(); }

  /// Get whether nested lifetimes are excluded from exclusivity checking.
  bool getIsNestedOriginsReadOnly() {
    return getBodyFnType().getIsNestedOriginsReadOnly();
  }

  /// Get whether the function establishes new interior origins.
  bool getDefinesInteriorOrigins() {
    return getBodyFnType().getDefinesInteriorOrigins();
  }

  /// Get the number of implicit origin decls this function type carries.
  size_t getNumImplicitOriginDecls() {
    return getBodyFnType().getNumImplicitOriginDecls();
  }

  Type getResultType() { return getBodyFnType().getResultType(); }

  /// Get the user result type of the signature.
  Type getUserResultType() { return getBodyFnType().getUserResultType(); }

  /// Get the user thrown type for a raising function.
  Type getUserThrownType() { return getBodyFnType().getUserThrownType(); }

  /// Returns true if the argument at this index is any vararg or a pack.
  bool isAnyVarArg(size_t index) { return getBodyFnType().isAnyVarArg(index); }

  /// Returns true if the argument at this index is a positional vararg.
  bool isPosVarArg(size_t index) { return getBodyFnType().isPosVarArg(index); }

  /// For a PosVarArg/PackVarArg, return the declared ArgConvention of the
  /// elements. For example: def x(mut *args: Int) is declared 'mut'.
  ArgConvention getVariadicConvention(size_t index) {
    return getBodyFnType().getVariadicConvention(index);
  }

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index) { return getBodyFnType().isKwVarArg(index); }

  /// Returns true if the argument at this index is a pack vararg.
  bool isPack(size_t index) { return getBodyFnType().isPack(index); }

  /// If the specified argument is a variadic list/pack, return the
  /// VariadicList/VariadicPack, stripping RefType, otherwise return null.
  Type getIfVariadicListOrPack(size_t index) {
    return getBodyFnType().getIfVariadicListOrPack(index);
  }

  /// Returns the index of the pack variadic arg, or std::nullopt if none.
  std::optional<size_t> findPackVarArgIndex() {
    return getBodyFnType().findPackVarArgIndex();
  }

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs() { return getBodyFnType().hasKwVarArgs(); }

  FunctionType substituteImplicitOriginsIntoValues(
      ArrayRef<TypedAttr> values,
      function_ref<InFlightDiagnostic()> emitError) {
    return getBodyFnType().substituteImplicitOriginsIntoValues(values,
                                                               emitError);
  }
};

//===----------------------------------------------------------------------===//
// FnTypeGeneratorType
//===----------------------------------------------------------------------===//
class FnTypeGeneratorType
    : public FnTypeWrapperGeneratorType<FnTypeGeneratorType,
                                        FuncTypeGeneratorType> {
public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  FnTypeGeneratorType(GeneratorType gen);
  FnTypeGeneratorType(FuncTypeGeneratorType gen);

  // CRTP for FnTypeWrapperGeneratorType
  FuncType getBodyFnType() { return getBody(); }
  PogListAttr getParamListAttrs();

  FuncType getBody();

  /// Reconstruct the generator using a list of named input parameters; these
  /// are prepended to the current signature and references are remapped to
  /// index references. If `paramNames` is non-empty, it must match
  /// `parentParams` in length. If `paramDefaults` is non-empty, it must match
  /// `parentParams` in length; a nullptr `TypedAttr` at index `i` means no
  /// default for that parameter. `bodyConstraints` are constraints enforced by
  /// the body that follows the prepended parameters.
  static FnTypeGeneratorType
  prependParams(FnTypeGeneratorType sig, ArrayRef<ParamDeclAttr> parentParams,
                ArrayRef<StringAttr> paramNames = {},
                ArrayRef<TypedAttr> paramDefaults = {},
                ArrayRef<ConstraintAttr> bodyConstraints = {});

  /// Return this signature with the specified capture lifetimes.
  FnTypeGeneratorType getWithCaptureOrigins(TypedAttr lifetimes);

  /// This method replaces direct uses of NAMED implicit origin declarations
  /// with index-based references corresponding to the signature.  lifetimeDecls
  /// specifies the names of the implicit origin decls.
  FnTypeGeneratorType
  replaceImplicitOriginsWithIndexes(ArrayRef<ParamDeclAttr> lifetimeDecls);

  /// This method replaces direct uses of NAMED implicit origin declarations
  /// with index-based references.  lifetimeDecls specifies the names of the
  /// implicit origin decls to replace.
  static Type replaceImplicitOriginsWithIndexes(
      Type type, ArrayRef<ParamDeclAttr> lifetimeDecls, size_t indexOffset = 0);

  static bool classof(FuncTypeGeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// FnLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

class FnLiteralTypeGeneratorType
    : public FnTypeWrapperGeneratorType<FnLiteralTypeGeneratorType,
                                        FuncLiteralTypeGeneratorType> {
public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  FnLiteralTypeGeneratorType(GeneratorType gen);
  FnLiteralTypeGeneratorType(FuncLiteralTypeGeneratorType gen);

  // CRTP for FnTypeWrapperGeneratorType
  FuncType getBodyFnType() { return getBody().getFuncType(); }
  PogListAttr getParamListAttrs();

  FuncLiteralType getBody();

  static bool classof(FuncLiteralTypeGeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// FnOrFnLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

// A simple wrapper around smart variant of FnTypeGeneratorType and
// FnLiteralTypeGeneratorType.
class FnOrFnLiteralTypeGeneratorType
    : public FnTypeWrapperGeneratorType<
          FnOrFnLiteralTypeGeneratorType,
          SmartVariant<FnTypeGeneratorType, FnLiteralTypeGeneratorType>> {

  using VariantT = FnTypeWrapperGeneratorType::BaseT;
  VariantT getAsVariant() const { return static_cast<VariantT>(*this); }

public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  /*implicit*/ FnOrFnLiteralTypeGeneratorType(FuncTypeGeneratorType gen)
      : FnTypeWrapperGeneratorType(sugarCast<FnTypeGeneratorType>(gen)) {}
  /*implicit*/ FnOrFnLiteralTypeGeneratorType(FuncLiteralTypeGeneratorType gen)
      : FnTypeWrapperGeneratorType(sugarCast<FnLiteralTypeGeneratorType>(gen)) {
  }

  static std::optional<FnOrFnLiteralTypeGeneratorType> tryGet(Type gen) {
    if (auto fnTypeGen = sugarDynCast<FnTypeGeneratorType>(gen))
      return fnTypeGen;
    if (auto fnLiteralGen = sugarDynCast<FnLiteralTypeGeneratorType>(gen))
      return fnLiteralGen;
    return std::nullopt;
  }
  static FnOrFnLiteralTypeGeneratorType get(Type gen) { return *tryGet(gen); }

  // Delegates to SmartVariant
  FnTypeGeneratorType getIfFnTypeGenerator() {
    return dyn_cast<FnTypeGeneratorType>(getAsVariant());
  }

  FnLiteralTypeGeneratorType getIfFnLiteralTypeGenerator() {
    return dyn_cast<FnLiteralTypeGeneratorType>(getAsVariant());
  }

  Type getAsType() const {
    if (auto ty = dyn_cast<FnTypeGeneratorType>(getAsVariant()))
      return ty;
    return cast<FnLiteralTypeGeneratorType>(getAsVariant());
  }

  // CRTP for FnTypeWrapperGeneratorType
  FuncType getBodyFnType() {
    if (auto fnGen = getIfFnTypeGenerator())
      return fnGen.getBodyFnType();
    return getIfFnLiteralTypeGenerator().getBodyFnType();
  }

  PogListAttr getParamListAttrs() {
    if (auto fnGen = getIfFnTypeGenerator())
      return fnGen.getParamListAttrs();
    return getIfFnLiteralTypeGenerator().getParamListAttrs();
  }

  // Debug Util.
  void dump() {
    if (auto fnGen = getIfFnTypeGenerator())
      fnGen.dump();

    getIfFnLiteralTypeGenerator().dump();
  }
};

//===----------------------------------------------------------------------===//
// MetaTypeOf
//===----------------------------------------------------------------------===//

template <typename T>
class MetaTypeOf : public MetaType {
public:
  using MetaType::MetaType;

  static MetaTypeOf<T> get(T type) {
    return llvm::cast<MetaTypeOf<T>>(MetaType::get(type));
  }

  T getType() const { return llvm::cast<T>(MetaType::getType()); }

  static bool classof(Type type) {
    auto metatype = llvm::dyn_cast<MetaType>(type);
    return metatype && llvm::isa_and_nonnull<T>(metatype.getType());
  }
};

class StructMetaType : public MetaTypeOf<LIT::StructType> {
private:
  using Base = MetaTypeOf<LIT::StructType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  StructMetaType(Base base) : Base(base) {}

  SymbolRefAttr getSymbol() const;
  TypeSignatureType getSignature() const;
  ArrayRef<TypedAttr> getParamValues() const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of param values. Only
  /// positions that are currently unbound can be updated.
  StructMetaType bindAll(ArrayRef<TypedAttr> values) const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of unbound parameters
  /// in the current param values list.
  StructMetaType bindUnbound(ArrayRef<TypedAttr> values) const;
};

class StructMetaMetaType : public MetaTypeOf<StructMetaType> {
private:
  using Base = MetaTypeOf<StructMetaType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  StructMetaMetaType(Base base) : Base(base) {}
  SymbolRefAttr getSymbol() const;
  TypeSignatureType getSignature() const;
  ArrayRef<TypedAttr> getParamValues() const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of param values. Only
  /// positions that are currently unbound can be updated.
  StructMetaMetaType bindAll(ArrayRef<TypedAttr> values) const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of unbound parameters
  /// in the current param values list.
  StructMetaMetaType bindUnbound(ArrayRef<TypedAttr> values) const;
};

class AnyTraitType : public MetaTypeOf<TraitType> {
private:
  using Base = MetaTypeOf<TraitType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  AnyTraitType(Base base) : Base(base) {}

  /// Return the trait type wrapped by this metatype.
  TraitType getTraitType() const { return getType(); }
};

class FnLiteralTypeGeneratorMetaType
    : public MetaTypeOf<FnLiteralTypeGeneratorType> {
private:
  using Base = MetaTypeOf<FnLiteralTypeGeneratorType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  FnLiteralTypeGeneratorMetaType(Base base) : Base(base) {}
};

class FnLiteralTypeGeneratorMetaMetaType
    : public MetaTypeOf<FnLiteralTypeGeneratorMetaType> {
private:
  using Base = MetaTypeOf<FnLiteralTypeGeneratorMetaType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  FnLiteralTypeGeneratorMetaMetaType(Base base) : Base(base) {}
};

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the user-defined result type of a signature, looking through
/// implicit memory results and stripping off the variant from error throwing
/// results if needed.
Type getSignatureUserResultType(FnOrFnLiteralTypeGeneratorType sigType,
                                ArrayRef<Type> argTypes, Type resultType);

/// If this specified operation is a call-like operation, return the
/// FnTypeGeneratorType for the callee, otherwise return null.
LIT::FnTypeGeneratorType getFnTypeFromCall(Operation &op);

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITTYPES_H
