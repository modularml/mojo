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
// Closure Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CLOSUREEMITTER_H
#define KGEN_MOJOPARSER_CLOSUREEMITTER_H

#include "ExprNodes.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/SharedState.h"
#include "StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {
class TypeCheckedFnSignature;
class TypeCheckedParamList;
struct AuxiliaryParameters;
using AliasSubstitutions = llvm::MapVector<mlir::StringAttr, TypedAttr>;
struct AdapteeParts {
  AliasSubstitutions aliasSubstitutions;
  DenseMap<StringAttr, TypedAttr> adapteeTypeMap;
  SmallVector<TypedAttr> fnLevelBindings;
  // flag to store when the callee returns in-register but the trait signature
  // expects a memory-only result.
  bool needsResultConversion = false;
};

/// Information about a closure parameter's external reference that needs
/// a where clause constraint. Contains the closure parameter along with the
/// name and type of the externalized reference.
struct ClosureExternalRef {
  /// The closure-typed parameter (e.g., "C" in `C: def(T) -> T`)
  ParamDeclAttr closureParam;

  /// The externalized name/type for the capture.
  StringAttr externalName;
  Type externalType;
};

class ClosureEmitter : public FunctionEmitter {
public:
  ClosureEmitter(SharedState &shared);

  /// Collect external parameter references from closure-typed parameters.
  ///
  /// For each parameter constrained by a closure trait, this examines the
  /// alias ops in the trait body. These aliases represent external parameter
  /// references from the outer scope where the closure was defined.
  ///
  /// Example:
  ///   def useIt[T: Coord, C: def(T) -> T](impl: C, arg: T)
  ///
  /// The closure trait for `C` will have an alias `T` in its body. This
  /// function collects that alias along with the closure param `C`.
  /// The caller can then generate: `where _type_is_eq_parse_time[T, C.T]()`
  ///
  /// \param closureParam The closure-typed parameter to examine
  /// \param refs Output vector for collected external references
  void collectClosureExternalRefs(ParamDeclAttr closureParam,
                                  SmallVectorImpl<ClosureExternalRef> &refs);

  /// Iterate over closure traits in a TraitType and invoke a callback for each.
  void processClosureTraits(TraitType traitType,
                            std::function<void(TraitDeclOp)> const &callback);

  /// Return the closure-defining trait declaration backing \p type, if \p type
  /// is a compiler-synthesized closure type. Looks through struct wrappers and
  /// reference types. Returns std::nullopt otherwise.
  static std::optional<TraitDeclOp> getClosureDecl(SharedState &shared,
                                                   Type type);

  /// Return true if \p type is a compiler-synthesized closure type.
  static bool isClosureType(SharedState &shared, Type type);

  /// Given a closure trait and "parameters", specialize the closure trait as
  /// though it had parameters to bind. That is, replace the aliases with the
  /// parameter bindings provided.
  TraitType getSpecializedClosureTrait(GeneratorType aliasGenerator,
                                       ArrayRef<TypedAttr> paramValues,
                                       ASTDecl &moduleDecl, SMLoc loc);

  /// Return true if \p type provably conforms to \p traitDecl.
  static bool provenConformsToTrait(ASTType type, ASTDecl *traitDecl,
                                    SharedState &shared,
                                    ArrayRef<ConstraintAttr> callerAssumptions);

  /// Generate a Parametric Closure Wrapper Struct, a struct that contains a
  /// parametric field. Both the field and the struct must conform to the
  /// associated closure trait characterized by the signature of the closure.
  ASTDecl *createClosureTrait(ASTDecl &moduleDecl,
                              FnTypeGeneratorType signatureType,
                              FnTypeGeneratorType key,
                              unsigned numPrependedCaptures,
                              SMLoc nestedFunctionOrTypeLocation);

  TraitType bindParamsToClosureTraitFromSig(IREmitter &emitter,
                                            ASTType traitType,
                                            FnTypeGeneratorType sig);

  struct PromotedClosureSelfArg {
    Type type;
    ArgConvention convention;
  };

  /// Move `nestedFnDecl` into `storageStructDecl` as a method and wire its
  /// captured values to the storage struct fields. Returns the method decl.
  ASTDecl *
  liftClosureIntoMethod(ASTDecl &nestedFnDecl, ASTDecl &storageStructDecl,
                        PromotedClosureSelfArg selfArg,
                        ArrayRef<StructDefFieldAttr> concreteFieldDecls,
                        ArrayRef<Value> concreteFieldCaptures,
                        ArrayRef<CaptureConvention> captureConventions,
                        ArrayRef<Type> selfBoundFieldTypes, Location location);

  /// Promote a closure decl into `targetParent`. nullptr → nearest FileModuleOp
  /// (thin/stateless promotions).
  ASTDecl *
  promoteClosure(ASTDecl &nestedFnDecl,
                 ArrayRef<ParamDeclAttr> prependedParams = {},
                 std::optional<PromotedClosureSelfArg> selfArg = std::nullopt,
                 std::optional<bool> capturingOverride = std::nullopt,
                 ASTDecl *targetParent = nullptr);

  /// Adapter overload for callsites that currently hold ParamDeclRefAttr.
  ASTDecl *
  promoteClosure(ASTDecl &nestedFnDecl,
                 ArrayRef<ParamDeclRefAttr> prependedParamRefs,
                 std::optional<PromotedClosureSelfArg> selfArg = std::nullopt,
                 std::optional<bool> capturingOverride = std::nullopt,
                 ASTDecl *targetParent = nullptr);

  Value emitClosure(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                    ArrayRef<Capture> captures, ASTDecl &traitDecl,
                    Location location, bool isCopyable,
                    FnTypeGeneratorType closureSig,
                    ArrayRef<ParamDeclRefAttr> paramCaptures);
  static ASTDecl *addCaptureValue(SharedState &shared, ASTDecl &closure,
                                  StringRef name, SMLoc location);

  static ASTDecl *addCaptureValue(ASTDecl &closure, SMLoc location,
                                  StringRef name, CaptureConvention capture,
                                  IREmitter &emitter,
                                  ASTDecl *signatureDecl = nullptr);
  /// Maps a raw closure signature to the canonical key and captured-parameter
  /// count used for closure-trait uniquing and name synthesis.
  static std::pair<FnTypeGeneratorType, unsigned>
  getClosureTraitKey(FnTypeGeneratorType rawSignature);
  ASTDecl *getOrCreateClosureTrait(FnTypeGeneratorType key,
                                   llvm::function_ref<ASTDecl *()> creation);
  /// Given a trait decl and a function signature, generate a struct that can
  /// wrap a function pointer to be used as a closure (`_PtrWrapper`).
  ASTDecl *createFnStructWrapper(ASTDecl &moduleDecl, ASTDecl &traitDecl,
                                 FnTypeGeneratorType signatureType,
                                 SMLoc location);

  /// Generate a stateless extension struct that extends one closure trait to
  /// a structurally compatible one.
  ASTDecl *createExtensionStruct(ASTDecl &moduleDecl, TraitDeclOp sourceTrait,
                                 TraitDeclOp targetTrait,
                                 ASTType sourceMetaType, SMLoc location);
  Type getConcreteClosureWrapperTypeForFnSymbol(ASTDecl &declScope, SMLoc loc,
                                                PValue fnPValue);

private:
  MLIRContext *ctx;

  // Cached attributes and types.
  StringAttr selfName, copyName;

  /// Underlying implementation of `augmentWitnessTablesToConformTo` and
  /// `isCompatibleWith`.
  LogicalResult checkStructCompatibility(ASTType structType, ASTDecl *traitDecl,
                                         bool emitRebind);

  /// Synthesize an adaptor function that rebinds the closure wrapper's
  /// __call__ from auxiliary parameters to trait aliases, then add the
  /// conformance witness table.
  void buildCallAdaptorAndAddWitness(StructDeclOp structDeclOp,
                                     ASTDecl &structDecl,
                                     TraitDeclOp traitDeclOp, FnOp traitCallFn,
                                     TypedAttr callee,
                                     const AdapteeParts &adapteeParts,
                                     ASTType selfTypeOverride = {});

public:
  /// If the wrapper conforms to a trait that is compatible with the desired
  /// trait, emit a rebind. For example, suppose we have a parameter P with a
  /// closure metatype defined by `def(x:Int) -> Int`. We should be able to bind
  /// a struct wrapper type W to P if W conforms to the trait `def(z:Int) ->
  /// Int`. This will require a rebind though because of the differences in
  /// argument names.
  LogicalResult augmentWitnessTablesToConformTo(ASTType structType,
                                                ASTDecl *closureTrait);

  /// Extend a source closure value/type constrained by one closure trait to a
  /// structurally compatible target closure trait, without changing the
  /// source's physical type.
  CValue createExtensionType(ASTDecl &fileModule, CValue sourceValue,
                             Type targetMetaType, TraitDeclOp targetTrait);

  /// Checks if the wrapper struct type conforms to a trait that is compatible
  /// with the desired trait.
  LogicalResult isCompatibleWith(ASTType structType, ASTDecl *traitDecl);

  /// Returns true if sourceTraitType can conform to targetTrait.
  LogicalResult isTraitCompatibleWith(ASTType sourceTraitType,
                                      TraitDeclOp targetTrait,
                                      ASTDecl *declScope = nullptr);

  /// One parent trait that a synthesized closure conforms to, named by symbol.
  struct ClosureParent {
    /// Resolves \p symbol's requirement up front; \p traitFnName is
    /// empty for marker traits, which have none.
    ClosureParent(SharedState &shared, TraitSymbolAttr symbol,
                  StringRef traitFnName, ClosureMethod closureMethod);

    ClosureParent(TraitSymbolAttr symbol, FnTypeGeneratorType traitFnSig,
                  StringAttr witnessName, ClosureMethod closureMethod)
        : symbol(symbol), closureMethod(closureMethod),
          witnessName(witnessName), signature(traitFnSig) {}

    TraitSymbolAttr getSymbol() const { return symbol; }
    StringAttr getWitnessName() const { return witnessName; }

    SymbolRefAttr getSymbolRef() const { return symbol.getSymbol(); }
    StringAttr getFlattenedName() const { return symbol.getFlattenedName(); }

    bool isEmpty() const { return closureMethod == ClosureMethod::NONE; }
    ClosureMethod getClosureMethod() const { return closureMethod; }

    TraitDeclOp getTrait(SharedState &shared) const;

    FnTypeGeneratorType getSignature() const { return signature; }

    InlineLevel getInlineLevel() const { return inlineLevel; }

  private:
    /// Symbol of the parent trait; its declaration is looked up from this.
    TraitSymbolAttr symbol;
    /// closure method tag corresponding to the method this parent represents.
    ClosureMethod closureMethod;
    /// The name used for building conformance table.
    StringAttr witnessName;
    /// The requirement function signature, null for marker trait.
    FnTypeGeneratorType signature;
    /// The requirement's inline level, which the synthesized witness inherits.
    InlineLevel inlineLevel = InlineLevel::Automatic;
  };

  /// This is `isEqualCanon` with one relaxation: parameters
  /// in the leading "before-`+`" region of the pog list (i.e.
  /// `PassingKind::Inferred`) are not user-bindable, so their names are
  /// arbitrary disambiguators and may differ.
  static bool isTypeRebindableTo(FuncTypeGeneratorType from,
                                 FuncTypeGeneratorType to);

  /// Bundles the IR artifacts produced by liftClosure.
  struct Closure {
    ASTDecl *structDecl;         ///< The closure storage struct.
    ASTDecl *promotedCallMethod; ///< The storage struct's `__call__` method.
    TypedAttr typeAttr;          ///< Bound closure storage struct type.
  };

private:
  /// Given a name, a list of builtin parent traits (like "Movable" for
  /// example), a location, and a populate method, return a trait declaration
  /// that inherits from the parent and contains the methods added to the
  /// function list populated by the populate method.
  std::pair<TraitDeclOp, ASTDecl *>
  createTraitOp(StringAttr name, SmallVector<ClosureParent> &parents,
                SMLoc nestedFunctionOrTypeLocation,
                llvm::function_ref<void(
                    ASTDecl &traitDecl,
                    DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
                    populateTrait);
  /// Construct the closure struct, lift the nested function into a method, and
  /// emit witness tables for all closure parents.
  Closure liftClosure(ASTDecl &moduleDecl, SMLoc smLoc,
                      SmallVector<ClosureParent> &closureParents,
                      SymbolRefAttr parentSymbolRef,
                      llvm::MapVector<StringRef, Type> const &aliases,
                      SmallVector<StructDefFieldAttr> &&fieldDecls,
                      SmallVector<Value> &&fieldCaptures,
                      SmallVector<CaptureConvention> &&fieldCaptureConventions,
                      SmallVector<ParamDeclAttr> &&allStructParams,
                      SmallVector<TypedAttr> &&structParamBindings,
                      StringAttr name, TypeConvention typeConvention,
                      SmallVector<Type> &&deviceCaptureFieldTypes,
                      bool capturesEncodable, ASTDecl &nestedFnDecl);

  /// Given the signature of a trait function, specialize it and add it to the
  /// struct as \p fnName.
  /// Returns
  /// (a) the new FnOp,
  /// (b) the parameters of the function minus the origins and remapped to
  /// reference struct parameters instead of indices
  /// (c) the result of the function, remapped to reference the struct
  /// parameters instead of indices.
  /// \p selfTypeOverride replaces the struct's own `Self` in the specialized
  /// signature; an extension's methods are written on the anchor it extends,
  /// not on the (stateless) extension struct.
  std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type> pushBackTraitFunctionImpl(
      FnTypeGeneratorType traitFnSignature, ASTDecl &structDecl, bool synthetic,
      StringAttr fnName, SpecialFunctionKind specialFnID,
      InlineLevel inlineLevel, bool redirectWitnessToImplParam = true,
      ASTType selfTypeOverride = {});
  struct DevicePassablePopulators {
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> isConvertible;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> isEncodable;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> toDeviceType;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> typeName;
    llvm::function_ref<TypedAttr()> deviceType;
  };
  /// Add DevicePassable conformance using callbacks that populate each
  /// interface member and return its witness.
  void
  addConformanceToDevicePassable(ASTDecl &structDecl,
                                 const DevicePassablePopulators &populators);
  /// Add DevicePassable conformance to closure storage (__storage), whose
  /// device_type reflects the device representations of its captures.
  void
  addStorageConformanceToDevicePassable(ASTDecl &structDecl,
                                        ArrayRef<Type> deviceCaptureFieldTypes,
                                        StringRef name);

  /// Look up a prelude trait used as a closure parent.
  ClosureParent getBuiltinParent(StringRef traitName, StringRef traitFnName,
                                 ClosureMethod closureMethod);

  /// We need to lazily resolve this builtin traits, since at the time closure
  /// emitter is constructed, they are not registered.
  ClosureParent getAnyParent() {
    if (anyParent.has_value())
      return *anyParent;
    return getBuiltinParent("AnyType", "", ClosureMethod::NONE);
  }
  ClosureParent getMoveParent() {
    if (moveParent.has_value())
      return *moveParent;
    return getBuiltinParent("Movable", "__init__", ClosureMethod::MOVE);
  }
  ClosureParent getDeinitableParent() {
    if (deinitableParent.has_value())
      return *deinitableParent;
    return getBuiltinParent("Deinitable", "__deinit__", ClosureMethod::DEL);
  }
  ClosureParent getRegisterPassableParent() {
    if (registerPassableParent.has_value())
      return *registerPassableParent;
    return getBuiltinParent("RegisterPassable", "", ClosureMethod::NONE);
  }
  ClosureParent getTrivialRegisterTypeParent() {
    if (trivialRegisterTypeParent.has_value())
      return *trivialRegisterTypeParent;
    return getBuiltinParent("TrivialRegisterPassable", "", ClosureMethod::NONE);
  }
  ClosureParent getCopyParent() {
    if (copyParent.has_value())
      return *copyParent;
    return getBuiltinParent("Copyable", "__init__", ClosureMethod::COPY);
  }
  ClosureParent getImplicitlyCopyableParent() {
    if (implicitlyCopyableParent.has_value())
      return *implicitlyCopyableParent;
    return getBuiltinParent("ImplicitlyCopyable", "", ClosureMethod::NONE);
  }

  /// AnyType is the base metatype for all types.
  std::optional<ClosureParent> anyParent;
  /// Movable trait is a parent of all closures.
  std::optional<ClosureParent> moveParent;
  /// Deinitable trait is a parent of all closures.
  std::optional<ClosureParent> deinitableParent;
  /// RegisterPassable marks the type as register passable.
  std::optional<ClosureParent> registerPassableParent;
  /// TrivialRegisterPassable marks the state as trivially register passable.
  std::optional<ClosureParent> trivialRegisterTypeParent;
  /// Copy trait is a parent of some closures.
  std::optional<ClosureParent> copyParent;
  /// ImplicitlyCopyable trait is a parent of some closures. It has no defining
  /// methods.
  std::optional<ClosureParent> implicitlyCopyableParent;
  /// Closure traits live in the top level module. This cache guards against
  /// emitting duplicates.
  DenseMap<Type, ASTDecl *> closureTraitCache;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
