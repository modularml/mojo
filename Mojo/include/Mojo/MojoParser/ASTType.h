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
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTTYPE_H
#define KGEN_MOJOPARSER_ASTTYPE_H

#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/Support/TriBool.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M {
class InflightDiag;

namespace KGEN {
class ParamDeclAttr;

} // namespace KGEN

namespace KGEN::LIT {
class ASTDecl;
class CValue;
template <typename ValueType>
struct ASTExprAnd;
class MojoInflightDiag;
enum class TypeConvention : uint32_t;
class RefType;
class RefPackType;
class SharedState;
class TraitType;

/// Context threaded through the recursive ASTType printer.
struct ASTTypePrinterContext;

/// This is a simple wrapper around an MLIR Type that provides helpful utilities
/// for working with our types, provides pretty printing in diagnostics, and
///
class ASTType {
public:
  /// The MLIR version of the type is conveniently accessible.
  Type mlirType;

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTType() {}

  // Implicitly convert to and from MLIR Type.
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTType(Type mlirType)
      : mlirType(mlirType) {}

  // Initialize an ASTType from a parameter expression of metatype type.
  ASTType(TypedAttr typeParamExpr);

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG operator Type() const {
    return mlirType;
  }

  /// ASTType is nullable.
  bool isNull() const { return !mlirType; }
  explicit operator bool() const { return !!mlirType; }
  bool operator!() const { return !mlirType; }

  /// Extract the metatype of this type, always return non-null is the ASTType
  /// itself is non-null.
  Type extractMetaType() const;

  /// If this is a user declared type, return the declaration that this came
  /// from, if it exists. If this is a raw MLIR type or a metatype, return null.
  ASTDecl *getDecl(SharedState &shared) const;

  /// If this is a parametric user defined type, return all parameter bindings
  /// on this reference to the type.  Note that this is potentially a partial
  /// binding set - incomplete bindings (missing bindings) are valid.
  ArrayRef<TypedAttr> getParamBindings() const;
  TypeSignatureType getSignature() const;

  /// Return this type with any parameter bindings removed.
  ASTType getWithoutParameters(SharedState &shared) const;

  /// Return true if this type has any unbound parameters.
  bool hasUnboundParameters() const;

  /// Return true if this ASTType is canonically equal (equal ignoring sugar) to
  /// the specified other type.
  bool isEqualCanon(ASTType other) const;

  /// Return true if this is the same as another ASTType are the same, or if
  /// they match when unbound parameters in the 'this' type are treated as
  /// the same as the corresponding parameter in the second type.
  ///    Foo[1] != Foo[2]   but  Bar[?, 1] == Bar[7, 1]
  bool isEqualAllowingUnbound(ASTType other, SharedState &shared) const;

  /// Return true if this is a None type.
  bool isNoneType() const;
  /// Return true if this is a TypeCheckError type.
  bool isTypeCheckErrorType() const;

  /// Return true if this type is a register-passable type that can be passed
  /// around and copied in SSA values instead of having to live in memory.
  ///
  /// The location specifies the location of the reference in case the use is
  /// invalid in this location.
  bool isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return the StructDeclOp::RegisterPassable enum for this type.
  TypeConvention getRegisterPassability(llvm::SMLoc loc,
                                        SharedState &shared) const;

  /// Return true if this type is RegisterPassable or if it is a generic type
  /// that could bind to a concrete RegisterPassable type.
  bool mightBeRegisterPassable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return the @__nonmaterializable decorator target for the type, or null if
  /// there is none.
  ASTType getNonmaterializableTarget(SharedState &shared) const;

  /// Return true if this type is a 'trivial' type, that is one that can be
  /// passed around by copying the bits, and whose destructor is a noop.
  bool isTrivial(llvm::SMLoc loc, SharedState &shared) const;

  /// Return the 'kind' of triviality this type possesses for the given
  /// trait: 'Deinitable', 'Copyable', or 'Movable'. Returns `yes` when
  /// provably trivial, `no` when provably non-trivial, and `unknown` when it
  /// can't be proven (e.g. an unfolded member).
  TriBool isSpecialFunctionTrivial(llvm::SMLoc loc, SpecialFunctionKind kind,
                                   SharedState &shared) const;

  /// Return true if this type is provably implicitly 'trivially' copyable, as
  /// defined by its '__copy_ctor_is_trivial' member and whether or not it
  /// adheres to the ImplicitlyCopyable trait. `scope` is required for the
  /// same reason as `isImplicitlyCopyable`'s.
  bool isProvablyImplicitlyTriviallyCopyable(llvm::SMLoc loc,
                                             SharedState &shared,
                                             ASTDecl &scope) const;
  /// Return true if this type is provably 'trivially' moveable, as defined by
  /// its '__move_ctor_is_trivial' member.
  bool isProvablyTriviallyMoveable(llvm::SMLoc loc, SharedState &shared) const;
  /// Return true if this type is provably 'trivially' deletable, as defined by
  /// its '__del__is_trivial' member.
  bool isProvablyTriviallyDeletable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type is implicitly/explicitly copyable, either because
  /// it is trivial or conforms to (Implicitly)Copyable trait. Note:
  /// this resolves the body of a struct type.
  ///
  /// `scope` is required so where-clause assumptions from the scope are used
  /// to prove conditional conformances (e.g., a synthesized copy ctor's
  /// where-clause can prove that an alias-resolved field type is Copyable).
  bool isCopyable(llvm::SMLoc loc, SharedState &shared, bool isImplicit,
                  ASTDecl &scope) const;
  bool isImplicitlyCopyable(llvm::SMLoc loc, SharedState &shared,
                            ASTDecl &scope) const;
  bool isExplicitlyCopyable(llvm::SMLoc loc, SharedState &shared,
                            ASTDecl &scope) const;

  /// Return true if this type is movable from its own type, either because it
  /// is trivial or has a move constructor from self. Note: this resolves the
  /// body of a struct type.
  ///
  /// `scope` is required so where-clause assumptions from the scope are used
  /// to prove conditional conformances.
  bool isMovable(llvm::SMLoc loc, SharedState &shared, ASTDecl &scope) const;

  /// Return true if this type is register passable or conforms to
  /// RegisterPassable trait.
  /// Note: this resolves the body of a struct type.
  bool isRegisterType(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type is trivial register passable
  /// or conforms to TrivialRegisterPassable trait.
  /// Note: this resolves the body of a struct type.
  bool isTrivialRegisterType(llvm::SMLoc loc, SharedState &shared) const;

  /// Check whether this type conforms to the specified trait, returning a
  /// 3-state result. This uses the concrete type's parameter bindings to
  /// evaluate any conditional trait conformances.
  ///
  /// When callerAssumptions is non-empty, where-clause assumptions are used
  /// to prove unfoldable conformance constraints (e.g., a where clause
  /// `AllWritable[*types]` can prove a Tuple's conditional Writable
  /// conformance).
  ///
  /// Returns:
  /// - `yes` if the type definitely conforms
  /// - `no` if the type definitely does not conform
  /// - `unknown` if conformance depends on constraints that cannot be
  ///   evaluated statically
  TriBool doesConformTo(TraitType trait, SharedState &shared,
                        ArrayRef<ConstraintAttr> callerAssumptions) const;

  /// Same, additionally collecting the problematic constraints so a diagnosing
  /// caller can name them.
  ConstraintResult
  doesConformToWithDetails(TraitType trait, SharedState &shared,
                           ArrayRef<ConstraintAttr> callerAssumptions) const;

  /// Given a standard trait like Copyable, look up the conformance.  On
  /// success, the ASTDecl of the trait itself is returned, it is otherwise
  /// null.
  std::pair<TriBool, ASTDecl *>
  conformsToBuiltinTrait(StringRef traitName, llvm::SMLoc loc,
                         SharedState &shared,
                         ArrayRef<ConstraintAttr> callerAssumptions) const;

  /// This returns true if the current type unconditionally conforms to the
  /// specified builtin trait, e.g. "Movable".
  bool provenConformsToBuiltinTrait(
      StringRef traitName, llvm::SMLoc loc, SharedState &shared,
      ArrayRef<ConstraintAttr> callerAssumptions) const;

  /// Given a reference, return the element as an ASTType.  This aborts
  /// if the current type isn't a reference.
  ASTType getReferenceElementType() const;

  struct ParameterListInfo {
    Type elementType;
    TypedAttr valueList;
  };
  /// Given a type of TypeList/ParameterList, return the element type and the
  /// values list that are bound to it. If it isn't a ParameterList/TypeList,
  /// return null.
  ParameterListInfo getParameterListInfo() const;

  /// Given a VariadicList, return parameters bound to it.
  struct VariadicListInfo {
    Type elementType;
    TypedAttr origin; // The !lit.origin of each element.
    bool isOwned;     // The VariadicList is "var".

    // The arguments passed into the variadic are always references. This
    // returns the type of the reference.
    RefType getElementRefType() const;
  };
  VariadicListInfo getVariadicListInfo() const;

  /// Return the RefPackType that corresponds to the VariadicPack instance.
  RefPackType getVariadicPackInfo(SharedState &shared) const;

  struct VariadicPackInfo {
    TypedAttr typeList;       // This is the !kgen.param_type of types.
    TypedAttr typeListStruct; // This is the TypeList for the types
    TypedAttr isOwned;        // This is the value of the is_owned parameter.
  };
  /// Decode the parameters list of VariadicPack.
  VariadicPackInfo getVariadicPackInfo() const;

  /// Given a variadic keyword dictionary type, return the dictionary's value
  /// type as an ASTType.
  ASTType getKwargsDictValueType() const;

  /// Given a variadic keyword dictionary reference type, return the
  /// dictionary's value type as an ASTType.
  ASTType getKwargsDictRefValueType() const;

  /// Returns the user-defined result type, looking through implicit memory
  /// results and stripping off the variant from error throwing results if
  /// needed.  This does NOT strip off the RefType for a 'ref[]' result.
  ASTType getSignatureUserResultType() const;

  /// If this type is parameterized, and if any of the parameters refer to a
  /// ParamIndexRefAttr, replace it with an UnboundAttr so parameter inference
  /// will infer it.
  ///
  /// This makes parameter inference sensitive to what to propagate vs infer.
  /// For example, if expectedType is known to be 'SIMD[uint8, 1]', then we can
  /// infer which constructor to use when the input is an IntLiteral.
  ///
  /// On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
  /// argument is an Int8, then we need the implicit conversion to infer the
  /// base element.  Our solution to this is to rip and replace parameters that
  /// contain unbound parameters, replacing them with UnboundAttr so inference
  /// can find them.
  ASTType getWithUnknownParametersReplaced(SharedState &shared) const;

  /// Return true if this type contains any origins that are unmaterializable
  /// from comptime to runtime.
  bool containsUnmaterializableOrigins(SharedState &shared) const;

  /// Convert this type to a human readable string representation so it can be
  /// printed out for diagnostics.  This may also be inserted into raw_ostream
  /// and diagnostics.
  std::string getAsString(ASTTypePrinterContext ctx) const;

  /// Print to standard error with newline after it, for use in a debugger.
  void dump() const;

  /// ASTType can be put into a PointerUnion, these are implementation details.
  void *getAsVoidPointer() const {
    return const_cast<void *>(mlirType.getAsOpaquePointer());
  }
  static ASTType getFromVoidPointer(void *ptr) {
    return ASTType(Type::getFromOpaquePointer(ptr));
  }

  /// Print the ASTType. If `ctx.shared` is set, prettier printing is used to
  /// print the type.
  void print(raw_ostream &os, ASTTypePrinterContext ctx) const;

  /// Print the specified parameter like we would in AST type printing.  When
  /// hasContextualType=true, this is being emitted into a context with a known
  /// contextual type, allowing us to elide implicit conversions and other
  /// noise.
  static void printParam(raw_ostream &os, TypedAttr param,
                         ASTTypePrinterContext ctx,
                         bool hasContextualType = false);

  /// Print the specified parameter like we would in an origin expression, works
  /// in an `origin_of(x)` body.  When `elideOriginOf` is true, the `origin_of(`
  /// and `)` will not be printed. This is important for 'ref' contexts.
  static void printOriginParam(raw_ostream &os, TypedAttr param,
                               SharedState *diagShared, bool elideOriginOf);
  /// Print the specified parameter like we would in a 'ref [x]' argument or
  /// result type, e.g. expanding origin sets.
  static void printRefOriginParam(raw_ostream &os, TypedAttr param,
                                  SharedState *diagShared);

  /// Get the specified parameter as a string.
  static std::string getParamAsString(TypedAttr param, SharedState *diagShared);

  /// Get the specified parameter as a string, works in an `origin_of(x)` body.
  static std::string getOriginAsString(TypedAttr param,
                                       SharedState *diagShared);

  /// Create and return a reference type with 'this' as the underlying element
  /// type an implicit origin reference with the specified arg name.
  RefType getRefForArgument(const Twine &argName, bool isMut);

  /// Extract a single field from a struct-typed value. Returns null if the
  /// field doesn't exist or the value isn't a struct type.
  static TypedAttr extractStructField(TypedAttr value, StringRef fieldName,
                                      llvm::SMLoc loc, SharedState &shared);

  /// Return the !lit.origin parameter if this type is a standard library
  /// `Origin` struct, otherwise return null.
  TypedAttr isOriginStruct() const;

  /// Given a parameter that is a !lit.origin or an Origin, return the
  /// underlying !lit.origin.  This returns null on failure.  This does not
  /// resolve the origin type, so it may be a !lit.origin or an Origin type.
  static TypedAttr extractOriginOf(TypedAttr value);

  /// Return true if this type is a singleton type. This is a type that has one
  /// value, e.g. a !lit.origin or a struct whose fields are all singleton
  /// types.
  bool isSingleton(SharedState &shared) const;
};
raw_ostream &operator<<(raw_ostream &os, ASTType type);

/// Context threaded through the recursive ASTType printer.
struct ASTTypePrinterContext {
  /// SharedState, for prettier diagnostic-style printing
  SharedState *shared = nullptr;
  /// An optional "self" type that, when matched against a sub-expression,
  /// prints `Self` in place of the fully-expanded form.
  ASTType selfType = {};

  /// Set to true to print closure trait signature.
  bool suppressThin = false;

  ASTTypePrinterContext() {}
  ASTTypePrinterContext(SharedState *shared) : shared(shared) {}
  ASTTypePrinterContext(SharedState *shared, ASTType selfType)
      : shared(shared), selfType(selfType) {}
};

} // namespace KGEN::LIT

} // namespace M

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::ASTType> {
public:
  using ASTType = M::KGEN::LIT::ASTType;
  static inline void *getAsVoidPointer(ASTType value) {
    return const_cast<void *>(value.getAsVoidPointer());
  }
  static inline ASTType getFromVoidPointer(void *pointer) {
    return ASTType::getFromVoidPointer(pointer);
  }
  enum {
    NumLowBitsAvailable = PointerLikeTypeTraits<void *>::NumLowBitsAvailable
  };
};

/// Cast from an (const) ASTType to a MLIR type.
template <typename T>
struct CastInfo<T, M::KGEN::LIT::ASTType>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::KGEN::LIT::ASTType,
                                     CastInfo<T, M::KGEN::LIT::ASTType>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::KGEN::LIT::ASTType type) {
    return type && T::classof(type.mlirType);
  }
  static T doCast(M::KGEN::LIT::ASTType type) { return cast<T>(type.mlirType); }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::ASTType>
    : public ConstStrippingForwardingCast<T, const M::KGEN::LIT::ASTType,
                                          CastInfo<T, M::KGEN::LIT::ASTType>> {
};
} // namespace llvm

#endif // KGEN_MOJOPARSER_ASTTYPE_H
