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
// Struct Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_STRUCTEMITTER_H
#define KGEN_MOJOPARSER_STRUCTEMITTER_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/SharedState.h"

namespace M::KGEN::LIT {

/// This struct bundles up analysis of the value members of a struct.
class ValueInfo {
public:
  FnOp del;      // __del__
  FnOp copyctor; // __init__(*, copy=)
  FnOp movector; // __init__(*, move=)

  static std::optional<ValueInfo> lookupExisting(ASTDecl &structDecl);

private:
  ValueInfo() {}
};

class FunctionEmitter : public SharedStateUser {
public:
  FunctionEmitter(SharedState &shared) : SharedStateUser(shared) {}

  /// Emit an empty function stub at the specified location. The block arguments
  /// are added to the body of the function but no ops are added to the body.
  /// `suffix` is appended to the mangled function name. This adds the
  /// declaration to `parent`.
  std::pair<FnOp, ASTDecl *> synthesizeFunction(
      ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
      ImplicitLocOpBuilder &builder, FnEffects fnEffects = FnEffects(),
      StringRef suffix = "", bool synthetic = true,
      InlineLevel inlineLevel = InlineLevel::Automatic);
};

class StructEmitter : public FunctionEmitter {
public:
  // The struct we're working on.
  ASTDecl &structDecl;
  StructDeclOp structDeclOp;

  StructEmitter(ASTDecl &structDecl);

  /// Generate empty stubs for the destructor, copy constructor, and move
  /// constructor on the declOp if they are eligible and do not already exist.
  ///
  /// A struct is eligible for a destructor if one of its fields has a
  /// destructor. It is possible that none of the fields of a struct have a
  /// destructor but that struct has an init that allocates heap memory. In this
  /// case set the forceGenerateDestructor flag to true to force destructor
  /// generation.
  std::optional<ValueInfo>
  addMissingValueMemberStubsToStruct(bool forceGenerateDestructor = false);

  /// Given a struct that has no explicitly defined `__del__` member, define a
  /// new one with an empty body. This allows the CheckLifetimes pass to insert
  /// field dels as needed, and makes sure that anything that refers to this
  /// struct properly runs its destructor.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause so that overload resolution rejects calls when the
  /// conditional conformance cannot be proven.
  FnOp synthesizeEmptyDtor(ConstraintAttr conformanceConstraint = {});
  /// Add an empty move/copy ctor stub for this struct, to be filled in later.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause so that overload resolution rejects calls when the
  /// conditional conformance cannot be proven.
  FnOp synthesizeEmptyMoveOrCopyInit(bool isMove,
                                     ConstraintAttr conformanceConstraint = {});
  /// Populate the function with a field by field copy. This will fail if the
  /// given function does not have the expected signature.
  LogicalResult populateMoveCopy(ASTDecl &fnDecl, bool isMove);

  /// Populate a trait function with its default implementation. Returns
  /// failure() if the meta-type to trait conversion fails; in this case the
  /// provided fnDecl is marked erroneous. Otherwise returns success().
  LogicalResult populateDefaultedTraitFunction(ASTDecl &fnDecl);

  /// Add a attribute initializer method for this struct with a body.
  FnOp synthesizeFieldwiseInit();

  /// This synthesizes an __init__ method that accepts values for every field of
  /// a struct, making it easy for external clients to initialize it.
  /// The `injectedFields` argument can be specified when creating an init
  /// method for memory-only types where not all fields are initialized, though
  /// this requires manual modification of the returned FnOp to initialize any
  /// omitted fields.
  FnOp synthesizeFieldwiseInit(ArrayRef<Type> argTypes,
                               ArrayRef<ArgConvention> argConventions,
                               PogListAttr argListAttrs,
                               // None or Self if register passable.
                               ASTType litReturnType);

  /// Create a FnOp within the scope of the given Struct. The body is not
  /// populated. `suffix` is appended to the mangled function name.
  std::pair<FnOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<ParamDeclAttr> params,
      PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);
  std::pair<FnOp, ASTDecl *> synthesizeMethodInStruct(
      StringRef name, ArrayRef<Type> argTypes,
      ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
      Type resultType,
      SpecialFunctionKind specialFnID = SpecialFunctionKind::kNormal,
      FnEffects fnEffects = FnEffects(), StringRef suffix = "",
      bool synthetic = true);

  /// Synthesize an unresolved alias into the struct with the specified name .
  ASTDecl *synthesizeUnresolvedAlias(StringRef name);
  TypedAttr populateSpecialFnIsTrivial(SpecialFunctionKind kind);

  /// Like synthesizeMethodInStruct but accepts higher-level signature
  /// information and extracts the necessary components internally. Also handles
  /// all the post-creation setup specific to default trait method wrappers.
  /// When \p conformanceConstraint is non-null, it is attached as a
  /// where-clause constraint on the synthesized function so that overload
  /// resolution rejects calls when the conditional conformance cannot be
  /// proven.
  FnOp synthesizeDefaultTraitMethodWrapper(
      ASTDecl &existingDecl, StringRef name,
      FnTypeGeneratorType wrapperSignature, FnOp traitFn, ASTDecl *traitFnDecl,
      ImplicitLocOpBuilder &builder, StringRef suffix = "",
      ConstraintAttr conformanceConstraint = {});
};

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
