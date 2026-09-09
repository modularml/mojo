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
// This pass checks value lifetime invariants, e.g. that variables are defined
// before use. This also inserts destructors for implicitly destroyed values.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/OriginTrackable.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/Threading.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;
using namespace LIT;
using llvm::BitVector;

static constexpr StringRef extraOriginUsesAttrName = ".mojo.extra.origin.uses";
static constexpr StringRef unusedMarkDestroyName =
    ".mojo.unused.mark_destroyed";
static constexpr StringRef selfPartiallyInitializedAttrName =
    ".mojo.self.partially.initialized";
static constexpr StringRef liveValueIdsAfterNoReturnCallAttrName =
    ".mojo.live.valueids.after.no.return.call";

namespace {
/// A simple raise set linked list. The elements are either BitVector (for dtor
/// analysis) or TrackedAndInteriorLiveness (for uninit pass).
template <typename ElementType>
struct RaiseSetEntry {
  StringAttr label;
  /// When analyzing the body of a try, this set element indicates what a
  /// 'raise' should produce based on its surrounding 'try's except block's
  /// expectation with the matching label.
  ElementType *raiseSet;
  // A linked list to the previous entry.
  RaiseSetEntry<ElementType> *prev;

  RaiseSetEntry<ElementType> *getMatchingRaiseSet(StringAttr label) {
    auto matchingSet = this;
    while (matchingSet && matchingSet->label != label)
      matchingSet = matchingSet->prev;
    return matchingSet;
  }
};
} // namespace

/// Create DebugInfo::DILocalVariableAttr if this VarDecl needs it.
/// `funcSpAttr` is the DISubprogramAttr of the surrounding function.
static DebugInfo::DILocalVariableAttr
createDebugVariableForVarDecl(VarDeclOp op,
                              DebugInfo::DISubprogramAttr funcSpAttr) {
  if (op.getKind() == VarDeclKind::Synthesized)
    return {};

  Location loc = op->getLoc();
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return {};

  auto localScope = DebugInfo::extractScopeFrom<DebugInfo::DILocalScopeAttr>(
      loc, DebugInfo::LocWalkPolicy::CalleePriority);
  if (!localScope)
    return {};

  // The source type is the decl type with ref unwrapped.
  Type elementType = op.getType().getElementType();

  // For ref bindings (e.g. loop variables), the VarDecl has two layers of ref:
  // the outer ref (VarDecl storage) and the inner ref (the reference to the
  // actual value). Unwrap the inner ref so the debugger shows the value type.
  if (op.getKind() == VarDeclKind::Ref)
    if (auto innerRef = dyn_cast<RefType>(elementType))
      elementType = innerRef.getElementType();

  auto sourceType = DebugInfo::DIUnresolvedMLIRType::get(elementType);
  auto varAttr = DebugInfo::DILocalVariableAttr::get(
      localScope, op.getNameAttr(), funcSpAttr.getFile(), fileLoc.getLine(),
      /*arg=*/0,
      /*alignInBits=*/0, sourceType, DebugInfo::DIFlags::Zero);

  return varAttr;
}

/// Inserts a DebugInfo::ValueOp for this block argument if necessary.
/// `funcSpAttr` is the DISubprogramAttr of the surrounding function `func`.
/// Returns the VarInfo of the inserted ValueOp.
static DebugInfo::DILocalVariableAttr
insertDebugVariableForArg(OpBuilder &builder, FnOp func, BlockArgument arg,
                          ArrayRef<PogMetadataAttr> pogList,
                          DebugInfo::DISubprogramAttr funcSpAttr) {
  // Skip synthesized args.
  if (arg.getArgNumber() >= pogList.size())
    return {};

  StringRef name = pogList[arg.getArgNumber()].getName();
  if (name.empty())
    return {};

  Location loc = arg.getLoc();
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return {};

  DebugInfo::DIExprAttr diExpr =
      DebugInfo::DIIRValueExprAttr::get(arg.getType());

  // If this argument has address, its needs an initial deref.
  ArgConvention convention =
      func.getFuncTypeGenerator().getArgConvention(arg.getArgNumber());
  if (hasAddress(convention)) {
    auto eltType = RefType::stripRefConvention(arg.getType(), convention);
    diExpr = DebugInfo::DIDerefExprAttr::get(diExpr, eltType);
  }

  DebugInfo::DIType sourceType =
      DebugInfo::DIUnresolvedMLIRType::get(diExpr.getType());
  DebugInfo::DIFlags flags = DebugInfo::DIFlags::Zero;
  if (convention == ArgConvention::ByRefError ||
      convention == ArgConvention::ByRefResult)
    flags = DebugInfo::DIFlags::Artificial;

  DebugInfo::DILocalVariableAttr varAttr = DebugInfo::DILocalVariableAttr::get(
      funcSpAttr, name, funcSpAttr.getFile(), fileLoc.getLine(),
      arg.getArgNumber() + 1,
      /*alignInBits=*/0, sourceType, flags);
  auto scopedLoc =
      FusedLoc::get(varAttr.getContext(), {loc}, varAttr.getScope());

  DebugInfo::ValueOp::create(builder, scopedLoc, arg, varAttr, diExpr);
  return varAttr;
}

//===----------------------------------------------------------------------===//
// WholeProgramState
//===----------------------------------------------------------------------===//

namespace {
struct StructInfo {
  /// This is the decl for the struct type, and is always present.
  LIT::StructDeclOp decl;
  /// This is the conformance information for Deinitable if it
  /// exists.
  ConformanceOp deinitable;
};
} // namespace

/// Module-level state shared across all worker threads in the CheckLifetimes
/// pass. The declaration maps are immutable after construction; the counter
/// uses atomic operations so no external lock is needed.
namespace {
struct WholeProgramState {
  WholeProgramState(Operation *module, std::vector<FnOp> &funcList);

  /// Immutable declaration maps, populated once before threading begins.
  DenseMap<SymbolRefAttr, StructInfo> structMap;
  DenseMap<SymbolRefAttr, FnOp> funcMap;
  DenseMap<SymbolRefAttr, LIT::TraitDeclOp> traitMap;

  // This is the Deinitable.__deinit__ trait member function.
  FnOp deinitableDtor;
};
} // namespace

/// Find all the functions and types in the module.
WholeProgramState::WholeProgramState(Operation *module,
                                     std::vector<FnOp> &funcList) {
  module->walk([&](Operation *op) {
    // Collect functions and nested functions.
    if (auto funcOp = dyn_cast<FnOp>(op)) {
      if (!funcOp.isOptionalSymbol())
        funcMap[getFullyResolvedSymbolRef(funcOp)] = funcOp;

      // We don't process external functions. They don't have a body to check.
      if (funcOp.isExternal())
        return;

      // Only top-level functions are parallel work items. A nested function
      // (an old closure) reads SSA values from the function it is nested in,
      // so it is processed within that top-level ancestor's task (see
      // runOnOperation) rather than concurrently with it.
      if (!funcOp->getParentOfType<FnOp>())
        funcList.emplace_back(funcOp);

      // Remember the Deinitable.__deinit__ member function.
      if (funcOp.getSpecialFunctionKind() == SpecialFunctionKind::kDeinit &&
          isa<TraitDeclOp>(funcOp->getParentOp()) &&
          cast<TraitDeclOp>(funcOp->getParentOp()).getSymName() == "Deinitable")
        deinitableDtor = funcOp;
    }

    // Collect structs.
    else if (auto structOp = dyn_cast<LIT::StructDeclOp>(op)) {
      // Find the conformance to Deinitable if it exists.
      ConformanceOp deinitableConformance;
      for (auto conformance : structOp.getFields().getOps<ConformanceOp>()) {
        if (conformance.getTraitSymbol()
                .getSymbol()
                .getLeafReference()
                .getValue() == "Deinitable") {
          deinitableConformance = conformance;
          break;
        }
      }
      structMap[getFullyResolvedSymbolRef(structOp)] = {structOp,
                                                        deinitableConformance};
    } else if (auto traitOp = dyn_cast<LIT::TraitDeclOp>(op)) {
      traitMap[getFullyResolvedSymbolRef(traitOp)] = traitOp;
    }
  });
}

//===----------------------------------------------------------------------===//
// TypeDeclInfo
//===----------------------------------------------------------------------===//

/// This is a wrapper to provide safer access to special members that are looked
/// up from traits.  Members can be in one of three states:
///   1) It may be unavailable, e.g. a destructor in a Linear type.
///   2) It may be available and meaningful.
///   3) It may be available but trivial, e.g. dtor for Int.
struct SpecialMemberInfo {
  bool isUnavailable() { return isa_and_nonnull<StringAttr>(storage); }

  /// If the special member is unavailable, return the message to emit if
  /// something tries to use it.
  StringAttr getMessageIfUnavailable() {
    if (auto stringAttr = dyn_cast<StringAttr>(storage))
      return stringAttr;
    return {};
  }

  /// This method can only be called when the member is available.  It returns
  /// the member function to call, or null if it is trivial.
  TypedAttr getMember() {
    assert(!isUnavailable() && "member is unavailable");
    return cast<TypedAttr>(storage);
  }

  static SpecialMemberInfo unavailable(StringAttr message) {
    return SpecialMemberInfo(message);
  }

  static SpecialMemberInfo available(TypedAttr member) {
    return SpecialMemberInfo(member);
  }

private:
  SpecialMemberInfo(TypedAttr storage) : storage(storage) {}
  /// This is the function to call if the special member is non-trivial or the
  /// message to print (a Stringattr). If the special member is trivial
  /// this is null.
  TypedAttr storage;
};

/// Per-thread information about struct declarations, used for field sensitive
/// analysis. Value tracking is completely field sensitive, tracking values at
/// the level of individual fields in their flattened representation. To do
/// this, we need an efficient mapping that tells us the number of (fully
/// flattened) fields in a struct.
struct TypeDeclInfo {
  TypeDeclInfo(const WholeProgramState *shared, Operation *module,
               mlir::LockedSymbolTableCollection *symtab)
      : shared(shared), evaluationContext(module, *symtab) {}

  /// Return the total number of flattened fields in the specified type.
  unsigned getNumFieldsInType(Type type) const;

  /// Return the start bit for a field with the specified name in the specified
  /// type.
  unsigned getFieldIndex(LIT::StructType type, StringAttr fieldName) const;
  int getFieldIndexOrInvalid(LIT::StructType type, StringAttr fieldName) const;

  /// Given a subfield bit index that indicates a stored field in the specified
  /// type, return the StructFieldOp of the accessed field, the first bit
  /// number covered by the subfield, and the total bits covered by the field.
  std::tuple<StructFieldOp, unsigned, unsigned>
  getFieldContaining(LIT::StructType type, unsigned bitIndex) const;

  /// Return the struct decl for the specified StructType.
  StructInfo getStructInfoForType(LIT::StructType type) const {
    auto it = shared->structMap.find(type.getSymbol());
    assert(it != shared->structMap.end() &&
           "reference to struct that wasn't declared");
    return it->second;
  }
  /// Return the trait decls for the specified TraitType (which may be a
  /// composition).
  SmallVector<LIT::TraitDeclOp, 2>
  getTraitDeclsForType(LIT::TraitType type) const {
    SmallVector<LIT::TraitDeclOp, 2> result;
    for (auto symbol : type.getSymbols()) {
      auto it = shared->traitMap.find(symbol.getSymbol());
      assert(it != shared->traitMap.end() &&
             "reference to trait that wasn't declared");
      result.push_back(it->second);
    }
    return result;
  }

  /// Return true if the specified type is RegisterPassableTrivial - no copy,
  /// move, or destructor members.
  bool isRegisterPassableTrivial(Type type) const;

  SpecialMemberInfo getDestructorForType(Type type, FnOp fnContext,
                                         Location loc) const;
  TypedAttr getMoveInitForType(Type type, Location loc) const;

  /// Return the function for a given symbol name if known.
  FnOp getFuncForSymbol(SymbolRefAttr symbolRef) const {
    auto it = shared->funcMap.find(symbolRef);
    return it != shared->funcMap.end() ? it->second : FnOp();
  }

  LIT::LITSymTabEvaluationContext *getEvaluationContext() const {
    return &evaluationContext;
  }

  /// Shared, immutable module-level state (declaration maps + anonymous-origin
  /// counter). Not owned here.
  const WholeProgramState *shared;

  /// Used for evaluating things like get_witness when we generate destructor
  /// calls.
  mutable LIT::LITSymTabEvaluationContext evaluationContext;

  /// This keeps track of the number of fields in the struct specified by the
  /// (fully flattened) symbol and parameters.
  mutable DenseMap<LIT::StructType, unsigned> numFields;

  /// A map from StructType and field name to index within the struct.  This
  /// isn't the field number, this is the number of recursively flattened
  /// fields until the start of the field.
  mutable DenseMap<std::pair<LIT::StructType, StringAttr>, unsigned>
      fieldIndices;
};

/// Return true if the specified type is RegisterPassableTrivial - no copy,
/// move, or destructor members.
bool TypeDeclInfo::isRegisterPassableTrivial(Type type) const {
  if (auto valueType = sugarDynCast<LIT::StructType>(type))
    return getStructInfoForType(valueType).decl.isRegisterPassableTrivial();

  // This is not trivial if it is a reference to a trait value.
  if (auto paramRef = sugarDynCast<ParamType>(type)) {
    if (auto trait = sugarDynCast<TraitType>(paramRef.getParam().getType())) {
      for (auto traitDecl : getTraitDeclsForType(trait))
        if (traitDecl.isRegisterPassableTrivial())
          return true;
      return false;
    }
  }

  // Other values of raw MLIR type are always trivial.
  return true;
}

static SpecialMemberInfo getSpecialMemberForType(
    Type type, const TypeDeclInfo *typeDecls,
    llvm::function_ref<SpecialMemberInfo(StructInfo)> getMember, Location loc) {
  auto valueType = sugarDynCast<LIT::StructType>(type);
  if (!valueType) // Values of raw MLIR type are trivial.
    return SpecialMemberInfo::available({});

  SpecialMemberInfo fnInfo =
      getMember(typeDecls->getStructInfoForType(valueType));
  if (fnInfo.isUnavailable())
    return fnInfo;

  // If there are parameters to the type, then the dtor will have those
  // parameters as well, substitute them in.
  if (valueType.getParamValues().empty() || !fnInfo.getMember())
    return fnInfo;
  TypedAttr member = fnInfo.getMember();
  TypedAttr result = BindParamsAttr::get(member.getContext(), member,
                                         valueType.getParamValues(),
                                         typeDecls->getEvaluationContext());
  return SpecialMemberInfo::available(result);
}

/// Append `bodyConstraints` to `assumptions`, converting each proposition's
/// parameter references from index to name form.
static void
appendNamedBodyConstraints(ArrayRef<ParamDeclAttr> params,
                           ArrayRef<ConstraintAttr> bodyConstraints,
                           SmallVectorImpl<ConstraintAttr> &assumptions) {
  ParamRefRemapper evaluator(params);

  for (ConstraintAttr constraint : bodyConstraints) {
    Attribute namedProposition = evaluator.replace(constraint);
    assert(namedProposition && "invalid constraint");
    assumptions.push_back(cast<ConstraintAttr>(namedProposition));
  }
}

/// Given a function context, return the constraints known from the context,
/// including trailing `where` clauses on enclosing struct/trait declarations.
static SmallVector<ConstraintAttr> getConstraintsFromContext(FnOp fnContext) {
  SmallVector<ConstraintAttr> assumptions;
  FnTypeGeneratorType fullSig = fnContext.getFullSignature();
  appendNamedBodyConstraints(
      fnContext.collectAllParams(/*includeImplOrigins=*/false),
      fullSig.getParamListAttrs().getBodyConstraints(), assumptions);
  return assumptions;
}

/// Refine a generic trait-bound type parameter using where-clause constraints
/// from the enclosing function signature.
static TraitType refineWithContextualWhereClauses(
    ParamType genericOfTraitType, FnOp fnContext,
    llvm::function_ref<TraitDeclOp(SymbolRefAttr)> traitDeclResolver) {
  // We know this is a parameter of trait type. If we can't refine it, then that
  // will be the default result:
  //    def foo[T: Trait1&Trait2](...)
  // result: T = lit.trait<Trait1, Trait2>
  TraitType curTrait =
      sugarCast<TraitType>(genericOfTraitType.getParam().getType());
  if (!curTrait)
    return curTrait;

  // Collect assumptions from signature metadata (see StructEmitter.cpp).
  SmallVector<ConstraintAttr> assumptions =
      getConstraintsFromContext(fnContext);
  if (assumptions.empty())
    return curTrait;

  TypedAttr paramRef = genericOfTraitType.getParam();
  TypedAttr lookupAttr = extractParamDeclRef(paramRef);
  if (!lookupAttr)
    lookupAttr = getCanonicalAttr(paramRef);

  TraitType refinedBound =
      getTraitBoundFromAssumptions(lookupAttr, assumptions, traitDeclResolver);
  if (!refinedBound)
    return curTrait;

  // Merge original bounds with refinements (see ExprNodes.cpp).
  SmallVector<TraitSymbolAttr> traitSymbols(curTrait.getSymbols());
  size_t origCount = traitSymbols.size();
  llvm::append_range(traitSymbols, refinedBound.getSymbols());
  sortAndDeduplicateTraitSymbols(traitSymbols);
  if (traitSymbols.size() == origCount)
    return curTrait;

  return TraitType::get(paramRef.getContext(), traitSymbols);
}

/// Given the RValue type for a value that needs to be destroyed, return the
/// destructor the invoke, or null if there is none.
SpecialMemberInfo TypeDeclInfo::getDestructorForType(Type type, FnOp fnContext,
                                                     Location loc) const {

  // If all the types in the trait composition are linear, then the trait
  // itself is linear.  If any of them is Deinitable, then the
  // whole thing is.
  auto getMessageIfTraitIsLinear = [&](ParamType generic,
                                       TraitType refinedTrait) -> StringAttr {
    // If all the types in the trait composition are linear, then the trait
    // itself is linear.  If any of them is Deinitable, then the
    // whole thing is.
    if (refinedTrait.getSymbols().empty())
      return StringAttr::get(generic.getContext(),
                             "type conforms to no traits");

    StringAttr message;
    for (TraitSymbolAttr symbol : llvm::reverse(refinedTrait.getSymbols())) {
      TraitDeclOp traitDecl = shared->traitMap.at(symbol.getSymbol());
      // If the trait has a linear type error message set, it means it does
      // not conform to Deinitable and is a linear type.
      if (auto errorMsg = traitDecl.getLinearTypeErrorMsgAttr()) {
        message = errorMsg;
        continue;
      }
      return {};
    }
    assert(message && "should have a message");
    return message;
  };

  if (auto generic = sugarDynCast<ParamType>(type)) {
    if (auto trait = sugarDynCast<TraitType>(generic.getParam().getType())) {
      TraitType refinedTrait = refineWithContextualWhereClauses(
          generic, fnContext, [&](SymbolRefAttr symbol) -> TraitDeclOp {
            return shared->traitMap.at(symbol);
          });

      // Check to see if this trait is linear.
      if (StringAttr message = getMessageIfTraitIsLinear(generic, refinedTrait))
        return SpecialMemberInfo::unavailable(message);

      // Otherwise, it is Deinitable, take the
      // Deinitable.__deinit__ member function and rebind Self to the
      // right type. If we didn't find Deinitable.__deinit__ (e.g. in
      // LSP cases) just assume everything is trivial.
      FnOp delFn = shared->deinitableDtor;
      if (!delFn)
        return SpecialMemberInfo::available({});

      // Bind Self to the right type.
      TraitDeclOp impDestroyTrait = cast<TraitDeclOp>(delFn->getParentOp());
      assert(impDestroyTrait.getParams().size() == 1 &&
             "Should have Self as a parameter");

      // Determine the result Self type.  We upcast the current
      // trait/composition up to Deinitable so we can set the
      // type.
      auto selfParam = UpcastAttr::get(impDestroyTrait.getParams()[0].getType(),
                                       generic.getParam());
      ParameterEvaluator evaluator(impDestroyTrait.getParams(), {selfParam});
      auto specSig = evaluator.getReboundType(delFn.getFuncTypeGenerator());

      // Okay, build the GetWitnessAttr.
      auto traitSymbol =
          TraitSymbolAttr::get(getFullyResolvedSymbolRef(impDestroyTrait));
      auto result = GetWitnessAttr::get(selfParam, traitSymbol,
                                        delFn.getSymNameAttr(), specSig);
      return SpecialMemberInfo::available(result);
    }
  }

  // Check if this specific instantiation of the struct type is
  // Deinitable by substituting the parameters into the where clause
  // constraint of Deinitable (if any).
  //
  // Returns nullopt if the trait isn't in the composition at all.
  auto isTypeDeinitable = [&](StructInfo info, Type structType) -> TriBool {
    TraitType canonTrait = info.decl.getCanonicalTrait();
    ArrayRef<TraitSymbolAttr> symbols = canonTrait.getSymbols();
    ArrayRef<ConstraintAttr> constraints = canonTrait.getConstraints();
    for (auto [i, symbol] : llvm::enumerate(symbols)) {
      if (symbol.getSymbol().getLeafReference() != "Deinitable")
        continue;
      if (i >= constraints.size())
        return TriBool::yes(); // Unconditional conformance.

      auto actualStructType = sugarCast<LIT::StructType>(structType);
      ParameterEvaluator evaluator(info.decl.getParams(),
                                   actualStructType.getParamValues());
      evaluator.setEvaluationContext(getEvaluationContext());
      TypedAttr conformanceCondition =
          evaluator.getReboundAttribute(constraints[i].getProposition());
      conformanceCondition = getCanonicalAttr(conformanceCondition);

      SmallVector<ConstraintAttr> assumptions =
          getConstraintsFromContext(fnContext);
      TypedAttr overallAssumption =
          SIMDAttr::getScalarBool(structType.getContext(), true);
      for (auto assumption : assumptions)
        overallAssumption = ParamOperatorAttr::get(
            POC::And, {overallAssumption, assumption.getProposition()});

      return isPropositionImplied(conformanceCondition, overallAssumption);
    }
    return TriBool::no();
  };

  auto getDestructor = [&](StructInfo info) -> SpecialMemberInfo {
    auto conformance = info.deinitable;
    // Determine conformance to Deinitable via the declared trait bound
    // of the struct type. This info is always available (in both LSP & normal
    // compile).
    TriBool isDeinitable = isTypeDeinitable(info, type);

    // - If the conformance condition is provably False, the type is NOT
    // Deinitable.
    // - If the conformance condition is unprovable, conservatively treat it as
    // NOT Deinitable. We can improve this error message in the future.
    if (!isDeinitable.isTrue()) {
      StringAttr message = info.decl.getLinearTypeErrorMsgAttr();
      assert(message && "should have a message");
      return SpecialMemberInfo::unavailable(message);
    }

    // After this point, we're confident the type is Deinitable.

    // If we didn't find a conformance op, we must be in LSP mode. Treat the
    // type as trivial since the exact dtor isn't important.
    if (!conformance)
      return SpecialMemberInfo::available({});

    // Fetch the destructor and trivial witness entries from the conformance op.
    TypedAttr dtorAttr;
    TypedAttr isTrivialAttr;
    for (WitnessOp witness : conformance.getOps<WitnessOp>()) {
      // ImplicitlyDestructable has two witnesses: Ignore del_is_trivial.
      if (witness.getName() == "__del__is_trivial") {
        assert(!isTrivialAttr && "Multiple __del__is_trivial witnesses found");
        isTrivialAttr = witness.getValue();
      } else {
        assert(witness.getName().starts_with("__deinit__(") &&
               "Unknown witness in Deinitable");
        assert(!dtorAttr && "Multiple dtors found in Deinitable");
        dtorAttr = witness.getValue();
      }
    }

    // If there is no destructor witness (e.g. conformance body not yet fully
    // resolved in LSP mode), treat this type as a trivial destructor.
    if (!dtorAttr)
      return SpecialMemberInfo::available({});
    assert(isTrivialAttr && "should have both witnesses");

    // Now that we found the conformance information, check to see if the
    // destructor is trivial. We have to substitute any "actual" parameters into
    // the __del__is_trivial witness to determine this.
    auto actualStructType = sugarCast<LIT::StructType>(type);

    ParameterEvaluator evaluator(info.decl.getParams(),
                                 actualStructType.getParamValues());
    evaluator.setEvaluationContext(getEvaluationContext());
    isTrivialAttr = evaluator.getReboundAttribute(isTrivialAttr);

    // The type of the trivial witness is Bool which wraps an i1.  If we can
    // prove that it is 1, then we can ignore this destructor.
    if (auto structAttr = sugarDynCast<LITStructAttr>(isTrivialAttr)) {
      if (structAttr.getValues().size() == 1) {
        if (auto boolAttr = sugarDynCast<SIMDAttr>(
                std::get<1>(structAttr.getValues().front()))) {
          if (boolAttr.getAsBool())
            return SpecialMemberInfo::available({}); // trivial!
        }
      }
    }

    // We will likely have a rebind to get rid of keyword argument and
    // convention differences; we don't care about this.
    dtorAttr = ParamOperatorAttr::stripRebind(dtorAttr);

    // The witness will refer to any struct parameters by name, we need to
    // rewrite them to indexes.  For example:
    //    struct Simple[a: Int]: def __deinit__(deinit self):
    // will have a __deinit__ witness of type
    //    !lit.generator<[1]("self": !lit.ref<!lit.struct<#Simple <:!Int a>>,
    //                      mut *[0,0]> deinit_mem, |) -> !kgen.none>
    // we want to rewrite this to:
    //    :!lit.generator<<"a": !Int, +>[1]("self":
    //    !lit.ref<!lit.struct<#Simple <:!Int *(0,0)>>, mut *[0,0]>
    //    deinit_mem) -> !kgen.none>
    //
    IndexRefRemapper remapper(info.decl.getInputParams(), 0);

    // Replace all the "a" type names with *(0, 0) within the generator body,
    // but move the type parameters+pogs onto the generator itself.
    auto fnType =
        remapper.replace(cast<GeneratorType>(dtorAttr.getType()).getBody());
    auto newType =
        GeneratorType::get(info.decl.getSignature().getParamTypes(), fnType,
                           info.decl.getSignature().getParamListAttrs());
    auto sym = cast<SymbolConstantAttr>(dtorAttr).getSymbol();
    return SpecialMemberInfo::available(
        SymbolConstantAttr::get(sym, newType, {}));
  };

  return getSpecialMemberForType(type, this, getDestructor, loc);
}

TypedAttr TypeDeclInfo::getMoveInitForType(Type type, Location loc) const {
  auto result = getSpecialMemberForType(
      type, this,
      [](StructInfo info) -> SpecialMemberInfo {
        if (auto moveInit = info.decl.getMoveInitAttr())
          return SpecialMemberInfo::available(moveInit);
        return SpecialMemberInfo::unavailable(StringAttr::get(
            info.decl.getContext(), "type does not conform to Movable"));
      },
      loc);

  if (!result.isUnavailable())
    return result.getMember();
  return {};
}

/// Return the total number of flattened fields in the specified type.
unsigned TypeDeclInfo::getNumFieldsInType(Type type) const {
  // We currently treat all non-struct types as being a single element, even
  // things like kgen.list containing struct types.
  auto structType = sugarDynCast<LIT::StructType>(type);
  if (!structType)
    return 1;

  // See if we've already looked this up, if so, just return the known value.
  auto it = numFields.find(structType);
  if (it != numFields.end())
    return it->second;

  // If not, we compute it recursively.  Structs cannot be infinitely deep, so
  // we can just do this recursively.
  auto smIt = shared->structMap.find(structType.getSymbol());
  assert(smIt != shared->structMap.end() && smIt->second.decl &&
         "reference to struct that wasn't declared");
  LIT::StructDeclOp decl = smIt->second.decl;

  // Initialize a parameter evaluator. We need to compute the resolved field
  // types to recursively compute the number of fields.
  ParameterEvaluator evaluator;
  for (auto [decl, value] :
       llvm::zip(decl.getInputParams(), structType.getParamValues()))
    evaluator.setDeclBinding(decl, value);

  size_t totalFields = 0;
  for (auto field : decl.getFieldDecls()) {
    fieldIndices[{structType, field.getNameAttr()}] = totalFields;
    totalFields +=
        getNumFieldsInType(evaluator.getReboundType(field.getType()));
  }

  // We always track an extra bit per struct.  On the outer level of a value
  // this tracks whether the object is fully constructed (not just field
  // constructed).  On individual fields, it tracks whether the field itself is
  // initialized or whether its subfields are initialized.  This also allows us
  // to support (sub)fields that have zero members soundly.
  ++totalFields;

  return numFields[structType] = totalFields;
}

/// Return the start bit for a field with the specified name in the specified
/// type, or -1 if the field isn't found.
int TypeDeclInfo::getFieldIndexOrInvalid(LIT::StructType type,
                                         StringAttr fieldName) const {
  auto it = fieldIndices.find({type, fieldName});
  return it == fieldIndices.end() ? -1 : it->second;
}

/// Return the start bit for a field with the specified name in the specified
/// type.
unsigned TypeDeclInfo::getFieldIndex(LIT::StructType type,
                                     StringAttr fieldName) const {
  int idx = getFieldIndexOrInvalid(type, fieldName);
  assert(idx >= 0 && "invalid field name for struct type");
  return unsigned(idx);
}

/// Given a subfield bit index that indicates a stored field in the specified
/// type, return the StructFieldOp of the accessed field, the first bit
/// number covered by the subfield, and the total bits covered by the field.
std::tuple<StructFieldOp, unsigned, unsigned>
TypeDeclInfo::getFieldContaining(LIT::StructType declRef,
                                 unsigned bitIndex) const {
  LIT::StructDeclOp decl = getStructInfoForType(declRef).decl;

  ParameterEvaluator evaluator(decl.getInputParams(), declRef.getParamValues());
  // Scan to find the field that contains this.
  unsigned startFieldIdx = 0;
  for (auto field : decl.getFieldDecls()) {
    // This range check is needed to handle zero-sized fields: they don't
    // contain a field even if they start at the beginning of it.
    Type reboundType = evaluator.getReboundType(field.getType());
    unsigned numSubFields = getNumFieldsInType(reboundType);
    if (startFieldIdx <= bitIndex && startFieldIdx + numSubFields > bitIndex)
      return {field, startFieldIdx, numSubFields};
    startFieldIdx += numSubFields;
  }

  llvm_unreachable("invalid index into struct field numbering");
}

/// Per-thread cache for the CheckLifetimes pass. One instance is created per
/// worker thread and reused across all functions processed by that thread,
/// so struct field-layout results and origin finder results accumulate across
/// functions rather than being recomputed for each one.
struct PerThreadCache {
  PerThreadCache(TypeDeclInfo &&typeDeclInfo)
      : typeDeclInfo(std::move(typeDeclInfo)), originFinder() {}

  /// Reset any shared state which could cause non-determinism in the
  /// compiler.
  void reset() { nextAnonOriginNumber = 0; }

  TypeDeclInfo typeDeclInfo;
  CachedOriginFinder originFinder;
  size_t nextAnonOriginNumber{0};
};

//===----------------------------------------------------------------------===//
// ValueInfo / ValueSet tracking
//===----------------------------------------------------------------------===//

// CheckLifetimes scans each function and identifies all the values that need
// to be tracked, in a field sensitive way.  The values are either directly
// modeled (this is for non-trivial register passable values, e.g. returned as
// owned from functions) or tracked as memory values. Memory values are tracked
// field sensitively, using some number of bits to (recursively) handle all the
// fields in the value.
namespace {
struct ValueRef;
struct ValueInfo {
  /// This is the declared value being tracked.  This can be null'd out if the
  /// value is completely removed.
  Value value;

  /// This indicates the (first, end] bitrange in the bit vector corresponding
  /// to this value.
  const unsigned startValueBit, endValueBit;

  /// True if this values starts out uninitialized at the beginning of its
  /// lifetime.
  const bool startsUninit;
  /// Enum indicating whether the value is initialized at function exit.
  const OriginTrackable::ExitInitState endInitState;

  /// True if this value lives in memory, not a RegisterPassable SSA value.
  const bool isIndirect;

  /// True if this is a byref_result argument for a self argument in an
  /// __init__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  const bool isFullObjectLiveOnEntry;

  /// This is true if the value had a use-before-initialization error diagnosed.
  bool hasErrorDiagnosed;

  /// This is true if the value was ever used.
  mutable bool isEverUsed;

  /// If this value needs to be tracked by debug info, this is the information
  /// about the source variable that created this value. Null otherwise.
  DebugInfo::DILocalVariableAttr debugVariable;

  /// Return true if this value contains the specified bit.
  bool contains(unsigned bitNo) const {
    return startValueBit <= bitNo && bitNo < endValueBit;
  }

  StringAttr getName() const {
    assert(value && "cannot get name of null entry");
    return OriginTrackable(value).name;
  }

  /// Return a ValueRef that covers this whole value.  The caller must provide
  /// the valueId.
  ValueRef getFullValueRef(unsigned valueId) const;

  /// Return true if this value should emit a warning on unused assignment.
  bool shouldWarnOnUnusedAssignment() const;

  /// Emit a new diagnostic error if this value has not yet been diagnosed.
  template <typename... Args>
  std::optional<InFlightDiagnostic> emitErrorIfNotDiagnosed(Args &&...args) {
    if (hasErrorDiagnosed)
      return std::nullopt;
    hasErrorDiagnosed = true;
    return mlir::emitError(std::forward<Args>(args)...);
  }
};

/// A ValueRef indicates a slice reference into the BitVector for all the
/// values.
struct ValueRef {
  /// This is the entry # for the ValueInfo for the overall value.
  unsigned valueId = 0;

  /// This is the (start, end] span of bits for the reference that we're
  /// tracking, which may be a subset of the overall value.
  unsigned startBit = ~0U, endBit = ~0U;

  /// This is true if this value reference is looking at the value indirectly,
  /// not as a RegisterPassable value in an SSA value.
  bool isIndirect = false;

  ValueRef() = default;
  ValueRef(unsigned valueId, unsigned startBit, unsigned endBit,
           bool isIndirect)
      : valueId(valueId), startBit(startBit), endBit(endBit),
        isIndirect(isIndirect) {}

  /// Allow use of a ValueRef in a boolean condition.
  operator bool() const { return valueId != 0; }

  unsigned getNumBits() const { return endBit - startBit; }

  bool operator==(ValueRef rhs) const {
    return startBit == rhs.startBit && endBit == rhs.endBit;
  }
  bool operator!=(ValueRef rhs) const { return !(*this == rhs); }

  /// Test if all the bits in the range are set in the specified BitVector.
  bool isAllPresent(const BitVector &bits) const {
    // BitVector doesn't have a more efficient method for this.  We could make
    // this more efficient for longer ranges if needed.
    for (size_t i = startBit, e = endBit; i != e; ++i)
      if (!bits[i])
        return false;
    return true;
  }

  /// Test if all the bits in the range are clear in the specified BitVector.
  bool isAllMissing(const BitVector &bits) const {
    // BitVector doesn't have a more efficient method for this.  We could make
    // this more efficient for longer ranges if needed.
    for (size_t i = startBit, e = endBit; i != e; ++i)
      if (bits[i])
        return false;
    return true;
  }

  /// Set the bits for this range to zero or one in the specified BitVector.
  void markBits(BitVector &bits, bool newValue) const {
    if (!valueId)
      return;
    if (newValue)
      bits.set(startBit, endBit);
    else
      bits.reset(startBit, endBit);
  }

  static Type getDereferencedType(Type sourceTy, bool isIndirect) {
    // If this is a direct value, use the type directly.
    return isIndirect ? cast<RefType>(sourceTy).getElementType() : sourceTy;
  }

  /// Return the type of the underlying value, looking through the reference
  /// type if indirect.
  Type getValueType(Value value) const {
    return getDereferencedType(value.getType(), isIndirect);
  }

  /// Given a field ref with fields, return a sub-field that starts at the
  /// specified bit offset and has the specified size.
  ValueRef getSubfield(unsigned offset, unsigned width) const {
    assert(startBit + offset + width <= endBit && "Not a valid subfield");
    return ValueRef(valueId, startBit + offset, startBit + offset + width,
                    isIndirect);
  }

  /// Return this ValueRef with the base offset subtracted off. This is useful
  /// when reasoning about a subfield inside another object without knowing the
  /// context.
  ValueRef getWithoutBaseOffset(unsigned offset) const {
    assert(startBit >= offset && "not offset by this base");
    return ValueRef(valueId, startBit - offset, endBit - offset, isIndirect);
  }
  ValueRef getWithBaseOffset(unsigned offset) const {
    return ValueRef(valueId, startBit + offset, endBit + offset, isIndirect);
  }

  /// Return true if this value ref is equal or a superset of the specified one.
  bool contains(ValueRef other) const {
    return startBit <= other.startBit && endBit >= other.endBit;
  }
};

/// Return a ValueRef that covers this whole value.  The caller must provide
/// the valueId.
ValueRef ValueInfo::getFullValueRef(unsigned valueId) const {
  return ValueRef{valueId, startValueBit, endValueBit, isIndirect};
}

/// Return true if this value should emit a warning on unused assignment.
bool ValueInfo::shouldWarnOnUnusedAssignment() const {
  // VarDecls can warn about unused assignments.
  if (auto varDecl = value.getDefiningOp<VarDeclOp>())
    // Don't warn about temporaries etc.
    return varDecl.shouldWarnAboutUnused();

  // 'var' / owned arguments should also warn.
  if (isa<BlockArgument>(value) &&
      endInitState == OriginTrackable::ExitInitState::EndsUninit)
    return true;

  return false;
}

/// This tracks the values in a function (including nested functions) that are
/// relevant for ownership - that needs to be tracked for uses without being
/// initialized, or that need a destructor to be run.
///
/// This tracks a /completely field sensitive/ view of the values under
/// consideration, including their nested fields in a flattened representation.
/// This gives us a fully precise view of the individual fields, and allows them
/// to be initialized and consumed in a piecewise way.
struct ValueSet {
  // This allows cached dominance computation within the current function.
  mlir::DominanceInfo domInfo;

  /// Per-thread analysis state cache. Stores the CachedOriginFinder and
  /// TypeDeclInfo.
  PerThreadCache &perThreadCache;

  /// When true, suppress debuginfo.kill markers for values whose type has
  /// no destructor.  This keeps trivially-destructible variables visible in
  /// the debugger through the end of their lexical scope at -O0.
  bool extendTrivialDebugLifetimes = false;

  /// Initialize the value set with one entry, so index #0 is always invalid and
  /// can be used as a sentinel, and so a null Value is always treated as
  /// untracked.
  ///
  /// This sentinel is also used by DestructorInsertion as a marker for
  /// "unreachable" code to avoid unnecessary meets.
  ValueSet(PerThreadCache &perThreadCache, FnOp func,
           bool extendTrivialDebugLifetimes = false);

  /// Return the function/closure we're analyzing.
  FnOp getFunc() const { return func; }

  /// Return the number of values we are tracking.
  MutableArrayRef<ValueInfo> getValueInfos() { return valueInfos; }
  ValueInfo &getValueInfo(size_t idx) { return valueInfos[idx]; }
  const ValueInfo &getValueInfo(size_t idx) const { return valueInfos[idx]; }

  /// Remove a tracked value from the valueset maps, and reset its ValueEntry to
  /// have a null Value.
  void eraseValueInfo(Value value);

  CachedOriginFinder &getOriginFinder() const {
    return perThreadCache.originFinder;
  }

  TypeDeclInfo &getTypeDeclInfo() const { return perThreadCache.typeDeclInfo; }

  size_t nextCounterValue() { return perThreadCache.nextAnonOriginNumber++; }

  /// Return a reference to the entire value with the specified ID.
  ValueRef getFullValueRef(unsigned valueId) const {
    auto &entry = valueInfos[valueId];
    entry.isEverUsed = true;
    return entry.getFullValueRef(valueId);
  }

  /// Given a origin attribute, return the value ref that defines it, and the
  /// known type of that value.  This can return a null type if we don't have
  /// field sensitive information.
  std::pair<ValueRef, Type> getValueRefAndTypeForOrigin(
      TypedAttr origin,
      SmallVectorImpl<InteriorOriginAttr> &interiorOrigins) const;

  /// Look up all the value refs that an access with the specified Value and
  /// dereference bit touch.
  SmallVector<ValueRef>
  getValueRefsForAccess(Value val, bool isDeref,
                        SmallVectorImpl<InteriorOriginAttr> &interiorOrigins);
  SmallVector<ValueRef>
  getValueRefsForOrigin(TypedAttr origin,
                        SmallVectorImpl<InteriorOriginAttr> &interiorOrigins);

  /// Given a tracked value that is being accessed by an operation, return
  /// the ValueRef for the object being tracked or null if untracked.
  ///
  /// 'isDeref' indicates that this is an indirect use of the specified value,
  /// which matters in the case of references.  When false, this is a use of a
  /// possibly-owned register value.
  ValueRef getDirectValueRef(Value value, bool isDeref) const;

  /// Return the total number of bits we need to track in the bitvector.
  unsigned getNumTotalBits() const {
    return !valueInfos.empty() ? valueInfos.back().endValueBit : 0;
  }

  /// Return true if this reference is to a trivial value that is not tracked
  /// for liveness.
  bool isTrivial(Type type, bool isIndirect) const {
    auto eltType = ValueRef::getDereferencedType(type, isIndirect);
    return perThreadCache.typeDeclInfo.isRegisterPassableTrivial(eltType);
  }

  bool isTrivial(Value value, bool isIndirect) const {
    return isTrivial(value.getType(), isIndirect);
  }

  /// Return true if we should suppress the debuginfo.kill for this value.
  /// At -O0, types without a destructor should stay visible in the debugger
  /// past their last use.  Without the kill marker, the -O0
  /// dbg.value→dbg.declare conversion gives them stable stack slots,
  /// keeping them visible for the whole scope.
  bool shouldSuppressDebugKill(const ValueInfo &info) const {
    if (!extendTrivialDebugLifetimes || !info.debugVariable)
      return false;
    if (!info.value)
      return false;
    // Get the underlying element type for the value.
    Type eltType;
    if (info.isIndirect) {
      if (auto refType = dyn_cast<RefType>(info.value.getType()))
        eltType = refType.getElementType();
      else
        return false;
    } else {
      eltType = info.value.getType();
    }
    SpecialMemberInfo dtorInfo =
        perThreadCache.typeDeclInfo.getDestructorForType(eltType, func,
                                                         info.value.getLoc());
    // Suppress kill when no non-trivial destructor will be emitted.
    return dtorInfo.isUnavailable() || !dtorInfo.getMember();
  }

  raw_ostream &printBV(const BitVector &bits, raw_ostream &os) const;
  LLVM_DUMP_METHOD void dumpBV(const BitVector &bits) const {
    auto &os = llvm::errs();
    printBV(bits, os) << "\n";
    os.flush();
  }

  LLVM_DUMP_METHOD void dump() const;
  void printFuncName(raw_ostream &os) const;

  // Get the location of the function we're scanning.
  Location getFuncLocation() { return func.getLoc(); }

private:
  /// This is the function we're analyzing.
  FnOp func;
  /// These are all of the value infos, indexed by ID #.
  SmallVector<ValueInfo> valueInfos;
  /// This is a lookup from SSA values to the thing they are referencing.
  DenseMap<Value, unsigned> valueInfoIndex;
  /// This is a mapping of origin attrs to the value index that defines them,
  /// these origins have sugar canonicalized away.
  DenseMap<TypedAttr, unsigned> originToValueIndex;

  /// Add a value to the set that we are tracking.  This includes:
  ///  * the MLIR representation for the value itself
  ///  * whether the value is a by-ref to the underlying logical value
  ///  * The bitrange it covers
  void addValue(Value val, const OriginTrackable &trackable,
                DebugInfo::DILocalVariableAttr debugVariable);
};
} // namespace

/// Initialize the value set with one entry, so index #0 is always invalid and
/// can be used as a sentinel, and so a null Value is always treated as
/// untracked.
///
/// This sentinel is also used by DestructorInsertion as a marker for
/// "unreachable" code to avoid unnecessary meets.
ValueSet::ValueSet(PerThreadCache &perThreadCache, FnOp func,
                   bool extendTrivialDebugLifetimes)
    : perThreadCache(perThreadCache),
      extendTrivialDebugLifetimes(extendTrivialDebugLifetimes), func(func) {
  addValue(Value(), OriginTrackable(Value()), DebugInfo::DILocalVariableAttr());

  // Check if the local variables of this function need debug info.
  DebugInfo::DISubprogramAttr funcSpAttr = func.getSubprogramScope();
  DebugInfo::DICompileUnitAttr compileUnit =
      funcSpAttr ? funcSpAttr.getCompileUnit() : nullptr;
  bool genDebugInfo = compileUnit && compileUnit.getEmissionKind() ==
                                         DebugInfo::EmissionKind::Full;

  func.getBodyRegion().walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) -> WalkResult {
        // Skip looking at nested functions, they are handled as separate
        // contexts.
        if (isa<FnOp>(op))
          return WalkResult::skip();

        // All the ops that define trackable values have a single result.
        if (op->getNumResults() == 1) {
          Value result = op->getResult(0);
          if (auto trackable = OriginTrackable(result)) {
            // Generate debug info for VarDecls if needed.
            DebugInfo::DILocalVariableAttr debugVariable;
            if (genDebugInfo) {
              if (auto varDecl = dyn_cast<VarDeclOp>(op)) {
                debugVariable =
                    createDebugVariableForVarDecl(varDecl, funcSpAttr);
              }
            }

            addValue(result, trackable, debugVariable);
          }
        }

        // If there are any regions, check the block arguments for arguments.
        for (auto &region : op->getRegions()) {
          for (auto &block : region)
            for (auto arg : block.getArguments())
              if (auto trackable = OriginTrackable(arg))
                addValue(arg, trackable, DebugInfo::DILocalVariableAttr());
        }

        return WalkResult::advance();
      });

  ArrayRef<PogMetadataAttr> pogList =
      func.getFuncTypeGenerator().getArgListAttrs().getPogs();
  OpBuilder debugBuilder =
      OpBuilder::atBlockBegin(&func.getBodyRegion().front());
  for (BlockArgument arg : func.getBodyRegion().front().getArguments()) {
    DebugInfo::DILocalVariableAttr debugVariable;
    if (genDebugInfo)
      debugVariable = insertDebugVariableForArg(debugBuilder, func, arg,
                                                pogList, funcSpAttr);
    if (auto trackable = OriginTrackable(arg))
      addValue(arg, trackable, debugVariable);
  }
}

/// Add a value to the set that we are tracking.  This includes:
///  * the MLIR representation for the value itself
///  * whether the value is a by-ref to the underlying logical value
///  * The bitrange it covers
void ValueSet::addValue(Value val, const OriginTrackable &trackable,
                        DebugInfo::DILocalVariableAttr debugVariable) {
  // Figure out how many bits to track for this value at the value if mem.
  unsigned numValueBits;
  TypedAttr valueOrigin;
  if (!val) {
    numValueBits = 1; // Nothing to do for the sentinel.
  } else if (trackable.isIndirect) {
    // This should be an assertion, but check softly to help IR clients.
    auto refType = dyn_cast<RefType>(val.getType());
    if (!refType) {
      mlir::emitError(val.getLoc())
          << "INTERNAL ERROR: trackable IR value of type " << val.getType()
          << " should have type '!lit.ref': " << val;
      return;
    }
    Type valType = refType.getElementType();
    numValueBits = getTypeDeclInfo().getNumFieldsInType(valType);

    // Remember the origin if not unknown.
    auto origin = getCanonicalAttr(refType.getOrigin());
    if (!isa<AnyOriginAttr>(origin))
      valueOrigin = origin;
  } else {
    // We don't track trivial values of register type.
    if (getTypeDeclInfo().isRegisterPassableTrivial(val.getType()))
      return;
    // We are only field sensitive for memory objects, not in-register values.
    numValueBits = 1;
  }
  unsigned firstValueBit = getNumTotalBits();

  // Record this information in our tables.
  valueInfoIndex[val] = valueInfos.size();
  if (valueOrigin)
    originToValueIndex[valueOrigin] = valueInfos.size();

  valueInfos.push_back(ValueInfo{
      val, firstValueBit, firstValueBit + numValueBits, trackable.startsUninit,
      trackable.endInitState, trackable.isIndirect,
      trackable.isFullObjectLiveOnEntry,
      /*hasErrorDiagnosed=*/false, /*isEverUsed=*/false, debugVariable});
}

raw_ostream &ValueSet::printBV(const BitVector &bv, raw_ostream &os) const {
  if (bv.size() != getNumTotalBits())
    return os << "WRONG LENGTH BIT VECTOR";

  os << '[';
  llvm::interleave(
      valueInfos,
      [&](const ValueInfo &vi) {
        for (size_t i = vi.startValueBit, e = vi.endValueBit; i != e; ++i)
          os << (bv.test(i) ? '1' : '0');
      },
      [&]() { os << ' '; });
  return os << ']';
}

void ValueSet::printFuncName(raw_ostream &os) const {
  os << "'" << const_cast<FnOp &>(func).getName() << "'";
}

void ValueSet::dump() const {
  auto &os = llvm::errs();
  os << "ValueSet with " << valueInfos.size() << " values for ";
  printFuncName(os);
  os << "\n";
  os << "  SI = startsInit, EI = endsInit, [*] = isIndirect";
  os << "  FL=isFullObjectLiveOnEntry, ERR = hadErrorDiag\n";

  for (auto [idx, info] : llvm::enumerate(valueInfos)) {
    os << "  #" << idx << " [" << info.startValueBit << ":" << info.endValueBit
       << ")";

    if (!info.startsUninit)
      os << " SI";
    switch (info.endInitState) {
    case OriginTrackable::EndsInit:
      break;
    case OriginTrackable::EndsUninit:
      os << " EI";
      break;
    case OriginTrackable::InitOnNormal:
      os << " NR";
      break;
    case OriginTrackable::InitOnError:
      os << " ER";
      break;
    }
    if (info.isIndirect)
      os << " [*]";
    if (info.isFullObjectLiveOnEntry)
      os << " FL";
    if (info.hasErrorDiagnosed)
      os << " ERR";
    os << "\t";

    if (!info.value) {
      os << "<<null sentinel>>\n";
      continue;
    }

    // If this is a function argument, be nice and include the name.
    if (auto bbArg = dyn_cast<BlockArgument>(info.value)) {
      if (auto fn = dyn_cast_or_null<FnOp>(bbArg.getOwner()->getParentOp()))
        os << fn.getFuncTypeGenerator().getArgName(bbArg.getArgNumber()) << " ";
    }

    os << info.value << "\n";
  }
  os.flush();
}

/// Remove a tracked value from the valueset maps, and reset its ValueEntry to
/// have a null Value.
void ValueSet::eraseValueInfo(Value value) {
  auto it = valueInfoIndex.find(value);
  assert(it != valueInfoIndex.end() && it->second && "not tracking this value");
  valueInfos[it->second].value = Value();
  valueInfoIndex.erase(it);
}

/// Given a origin attribute, return the value ref that defines it, and the
/// known type of that value.  This can return a null type if we don't have
/// field sensitive information.
std::pair<ValueRef, Type> ValueSet::getValueRefAndTypeForOrigin(
    TypedAttr origin,
    SmallVectorImpl<InteriorOriginAttr> &interiorOrigins) const {
  // The mutability of the origin access doesn't affect what ValueRef is
  // accessed.
  origin = OriginMutCastAttr::strip(getCanonicalAttr(origin));

  // If the origin has one or more field specifiers like 'a.x.y.z', find
  // the ValueRef for the base and then refine it.
  if (auto field = sugarDynCast<OriginFieldAttr>(origin)) {
    auto [valueRef, type] =
        getValueRefAndTypeForOrigin(field.getBase(), interiorOrigins);
    // If we don't have field sensitive information then we cannot refine the
    // origin.  This also handles the null valueRef case.
    if (!type)
      return {valueRef, type};

    assert(valueRef.isIndirect && "Cannot field refine SSA value access");
    auto fieldName = field.getField();

    auto containerType = sugarDynCast<LIT::StructType>(type);
    if (!containerType)
      return {valueRef, Type()};
    int fieldOffset =
        getTypeDeclInfo().getFieldIndexOrInvalid(containerType, fieldName);
    if (fieldOffset == -1)
      return {valueRef, Type()};

    // Figure out the declared type of the field.
    auto [fieldDecl, _, numFieldBits] =
        getTypeDeclInfo().getFieldContaining(containerType, fieldOffset);
    assert(fieldDecl.getNameAttr() == fieldName && "index/name mismatch");

    // Refine the ValueRef and type.
    return {valueRef.getSubfield(fieldOffset, numFieldBits),
            fieldDecl.getType()};
  }

  // If this is an interior origin like `a.field["name"]`, then the interior
  // origin isn't directly tracked, but the base origin is being treated as
  // being accessed.
  if (auto interior = sugarDynCast<InteriorOriginAttr>(origin)) {
    interiorOrigins.push_back(interior);
    auto [valueRef, _] =
        getValueRefAndTypeForOrigin(interior.getBase(), interiorOrigins);
    // Don't pass the type up; the indirected type isn't the same as the field
    // we may be accessed from.
    return {valueRef, {}};
  }

  // If this is an subtree origin like `a.field~`, then we treat this like an
  // access to the base `a.field`.
  if (auto subtree = sugarDynCast<OriginSubtreeAttr>(origin)) {
    auto [valueRef, _] =
        getValueRefAndTypeForOrigin(subtree.getBase(), interiorOrigins);
    // Don't pass the type up; the base type will have fields erased.
    return {valueRef, Type()};
  }

  // Otherwise look up the base origin value.
  auto it = originToValueIndex.find(origin);
  if (it == originToValueIndex.end())
    return {};

  const auto &entry = valueInfos[it->second];
  assert(entry.isIndirect);
  return {entry.getFullValueRef(it->second),
          cast<RefType>(entry.value.getType()).getElementType()};
}

/// Given a pointer that is being accessed indirectly by an operation, return
/// the value number being referenced, or zero if not tracked.
///
/// 'isDeref' indicates that this is an indirect use of the specified value,
/// which matters in the case of references.  When false, this is a use of a
/// possibly-owned register value.
ValueRef ValueSet::getDirectValueRef(Value value, bool isDeref) const {
  // If the value is deref, it must have reference type.
  assert((!isDeref || isa<RefType>(value.getType())) &&
         "only references are dereferenceable!");

  // If this is testing a reference value (not the dereference value) then it is
  // ignored: references can be passed around and used with the contents being
  // liveness tracked, the ultimate accesses are what matter.
  if (!isDeref && isa<RefType>(value.getType()))
    return {};

  // If this is a RebindOp get the underlying ref.
  value = RefImmutOp::stripRebinds(value);

  // If this is a value we're tracking, return it.
  auto it = valueInfoIndex.find(value);
  if (it != valueInfoIndex.end())
    return getFullValueRef(it->second);

  // If this is a GER, check the base and focus in on a field of it.
  if (auto structGER = value.getDefiningOp<RefStructGEROp>()) {
    // For index access, we can't determine field offset without concrete index.
    // Treat as opaque access to the full container.
    if (!structGER.usesFieldAccess())
      return {};

    ValueRef baseVal = getDirectValueRef(structGER.getContainer(), isDeref);
    if (!baseVal || !baseVal.isIndirect)
      return {};

    // Figure out what subset of elements we have indexed to.
    auto containerType = structGER.getContainer().getType().getElementType();
    unsigned fieldOffset = getTypeDeclInfo().getFieldIndex(
        sugarCast<LIT::StructType>(containerType), structGER.getFieldAttr());
    unsigned startBit = baseVal.startBit + fieldOffset;
    auto resultType = structGER.getType().getElementType();
    return ValueRef{baseVal.valueId, startBit,
                    startBit + getTypeDeclInfo().getNumFieldsInType(resultType),
                    /*isIndirect=*/true};
  }

  if (auto load = value.getDefiningOp<RefLoadOp>())
    if (auto valueRef = getDirectValueRef(load.getRef(), /*isDeref=*/true)) {
      if (valueRef.isIndirect) {
        // The parser doesn't emit all the lifetime stuff for trivial types,
        // so don't track them either.
        if (isTrivial(load, /*isDeref=*/false))
          return {};

        valueRef.isIndirect = false;
        return valueRef;
      }
    }

  // Otherwise, we don't know what this is.
  return ValueRef();
}

/// Look up all the value refs that an access to the specified origin could
/// touch.
SmallVector<ValueRef> ValueSet::getValueRefsForOrigin(
    TypedAttr origin, SmallVectorImpl<InteriorOriginAttr> &interiorOrigins) {
  SmallVector<ValueRef> result;

  // Look through imm cast and unions to find the underlying attrs.
  processOriginUnionElts(getCanonicalAttr(origin), [&](TypedAttr raw) {
    auto [valueRef, _] = getValueRefAndTypeForOrigin(raw, interiorOrigins);
    if (valueRef) {
      result.push_back(valueRef);
      valueInfos[valueRef.valueId].isEverUsed = true;
    }
  });

  return result;
}

/// Look up all the value refs that an access with the specified Value and
/// dereference bit touch.
SmallVector<ValueRef> ValueSet::getValueRefsForAccess(
    Value value, bool isDeref,
    SmallVectorImpl<InteriorOriginAttr> &interiorOrigins) {
  // If this is a direct reference to a value, return field sensitive info.
  if (ValueRef valueRef = getDirectValueRef(value, isDeref)) {
    SmallVector<ValueRef> result;
    result.push_back(valueRef);
    return result;
  }

  // Otherwise, if indirect, this is an reference to one or more
  // origin-tracked values, figure out what they are.
  if (isDeref)
    return getValueRefsForOrigin(cast<RefType>(value.getType()).getOrigin(),
                                 interiorOrigins);

  // Otherwise it is a trivial or untracked value.
  return {};
}

//===----------------------------------------------------------------------===//
// InteriorOrigin tracking
//===----------------------------------------------------------------------===//

namespace {
/// Liveness state for tracked values and interior origins at a program point.
struct TrackedAndInteriorLiveness {
  /// Bitvector of tracked value/field liveness.  Slot #0 is a reachability
  /// sentinel used by control-flow merge.
  BitVector tracked;

  /// For each interior origin, this contains one of three states:
  /// - Never initialized: False + the operation is null.
  /// -       Invalidated: False + the operation that invalidated it.
  /// -             Valid: True + the first operation that defined it.
  using InteriorEntry = llvm::PointerIntPair<Operation *, 1, bool>;
  SmallVector<InteriorEntry> interior;

  TrackedAndInteriorLiveness(size_t numTrackedBits, size_t numInteriorBits)
      : tracked(numTrackedBits), interior(numInteriorBits) {}

  static TrackedAndInteriorLiveness
  getEmptyWithMatchingSize(const TrackedAndInteriorLiveness &other) {
    return TrackedAndInteriorLiveness(other.tracked.size(),
                                      other.interior.size());
  }

  /// Return the # of directly tracked bits and the # of interior origins that
  /// are live at this point. This is used for loop convergence checking.
  std::pair<size_t, size_t> countNumLive() const {
    size_t numInteriorLive = 0;
    for (size_t i = 0, e = interior.size(); i != e; ++i) {
      if (interior[i].getInt())
        numInteriorLive++;
    }
    return {tracked.count(), numInteriorLive};
  }

  bool isReachable() const { return tracked[0]; }
  void markReachable(bool value) { tracked[0] = value; }

  /// Merge \p other into this state at a control-flow join.
  void mergeWith(const TrackedAndInteriorLiveness &other,
                 mlir::DominanceInfo &domInfo);

  void swap(TrackedAndInteriorLiveness &other) {
    tracked.swap(other.tracked);
    interior.swap(other.interior);
  }

  /// Print tracked value liveness and, when present, interior-origin validity.
  raw_ostream &print(const ValueSet &valueSet, raw_ostream &os) const {
    valueSet.printBV(tracked, os);
    if (!interior.empty()) {
      os << ", interior = {";
      for (size_t i = 0, e = interior.size(); i != e; ++i) {
        if (interior[i].getInt())
          os << " #" << i;
      }
      os << " }";
    }
    return os;
  }

  LLVM_DUMP_METHOD void dump(const ValueSet &valueSet) const {
    print(valueSet, llvm::errs());
  }
};
} // namespace

/// Lift \p op toward \p other by climbing parent ops until \p op's region is an
/// ancestor of \p other's region.
static Operation *liftToCommonRegion(Operation *op, Operation *other) {
  Region *otherRegion = other->getParentRegion();
  while (!op->getParentRegion()->isAncestor(otherRegion))
    op = op->getParentOp();
  return op;
}

/// Given two operations indicating interior-origin validity, pick the defining
/// operation that remains valid after a control-flow merge with the shortest
/// lifetime. Same-block cases pick the later operation; sibling regions are
/// lifted to their nearest common ancestor.
static Operation *findNearestCommonAncestor(Operation *op1, Operation *op2,
                                            mlir::DominanceInfo &domInfo) {
  if (op1 == op2)
    return op1;
  if (op1->getBlock() == op2->getBlock())
    return domInfo.dominates(op1, op2) ? op2 : op1;

  Operation *op1Common = liftToCommonRegion(op1, op2);
  Operation *op2Common = liftToCommonRegion(op2, op1);
  return domInfo.dominates(op1Common, op2Common) ? op2Common : op1Common;
}

/// Merge \p other into this state at a control-flow join.
void TrackedAndInteriorLiveness::mergeWith(
    const TrackedAndInteriorLiveness &other, mlir::DominanceInfo &domInfo) {
  if (!other.isReachable())
    return; // 'other' isn't reachable, so just use 'this'.
  if (!isReachable()) {
    // 'this' isn't reachable, so just use 'other'.
    tracked = other.tracked;
    interior = other.interior;
    return;
  }

  // Both are reachable, so merge them.
  tracked &= other.tracked;
  for (size_t i = 0, e = interior.size(); i != e; ++i) {
    auto &entry = interior[i];
    auto &otherEntry = other.interior[i];
    if (entry.getInt() && !otherEntry.getInt()) {
      // We're invalided if the other one is.
      entry = otherEntry;
    } else if (!entry.getPointer() && otherEntry.getPointer()) {
      entry.setPointer(otherEntry.getPointer());
    } else if (entry.getInt() && otherEntry.getInt()) {
      // If both are live, we should pick the common ancestor.
      entry.setPointer(findNearestCommonAncestor(
          entry.getPointer(), otherEntry.getPointer(), domInfo));
    }
  }
}

// These data structures keep track of all of the interior origins found in a
// function. This allows us to use bitvector dataflow to push around their
// liveness and allows us to understand what origins need to be invalidated when
// something with an interior origin is mutated.
//
// For example consider:
//    ref r1 = a.list[].first[].y
//    ref r2 = a.list[].second.field[].z
//    ref r3 = a.list~
//
// The "r1" reference must be invalided when any of "a", "a.list", or
// "a.list[].first" are mutated. The "r2" reference must be invalidated when any
// of "a", "a.list", "a.list[].second", or "a.list[].second.field" are mutated.
//
// Subtree origins are defined as a synthetic interior origin (a.list[<magic>])
// but mutations to them need to be treated like mutations to the base origin,
// because "a.list" converts to "a.list~". Furthermore, they need to be
// invalidated when any interior origins derived from the base are mutated,
// because it may alias them as well. This means the "r3" reference must be
// invalidated when any of these origins are mutated.
//
// In order to make this efficient, we keep invert the origin representation to
// hold an 'interiorOriginInvalidationMap' indicating what to invalidate when
// an origin is mutated that has derived interior origins. In this case, it has:
//
//    a.list[].first => { a.list[].first[], a.list~ }
//    a.list[].second => { a.list[].second.field[], a.list~ }
//    a.list[].second.field => { a.list[].second.field[], a.list~ }
//    a.list[] => { a.list[].first[], a.list[].second.field[] }
//    a.list~ => { same as a.list[] + a.list~ itself }
//    a.list.otherfield => { a.list~ }
//    a.list => { a.list[], a.list[].first[], a.list[].second.field[], a.list~ }
//    a => { all the interiors + a.list~ }
//
// Note that the trailing field sensitivity information is not tracked since 'y'
// and 'z' aren't special - all subfields of the final interior origin are
// invalidated (or not) together.
namespace {
class InteriorOriginTracker {
public:
  InteriorOriginTracker(PerThreadCache &perThreadCache, FnOp func);
  InteriorOriginTracker(const InteriorOriginTracker &existing) = delete;

  LLVM_DUMP_METHOD void dump() const;

  /// Print the interior origin IDs set as `{ #N ... }`.
  static void printBV(const BitVector &interiorOriginBits,
                      llvm::raw_ostream &os);

  /// This returns the number of distinct interior origins in the function.
  size_t getNumInteriorOrigins() const { return nextInteriorOriginID; }

  /// Given an InteriorOrigin, return its ID.
  size_t getInteriorOriginID(InteriorOriginAttr o) const {
    auto it = interiorOriginID.find(o);
    // If we're going to explode, produce some debris to help debug.
    if (!(it != interiorOriginID.end() && it->second < nextInteriorOriginID)) {
      llvm::errs() << "\n-----\n";
      o.dump();
      llvm::errs() << "\n-----\n";
      dump();
      llvm::errs() << "\n-----\n";
      theFunc->dumpPretty();
    }
    assert(it != interiorOriginID.end() && it->second < nextInteriorOriginID &&
           "Unknown interior origin");
    return it->second;
  }

  /// Mark the given interior origin as live in the given bitvector.
  void markInteriorOriginLive(InteriorOriginAttr o,
                              TrackedAndInteriorLiveness &liveness,
                              Operation &op,
                              const mlir::DominanceInfo &domInfo);

  bool originHasInteriorOrigins(TypedAttr origin) const {
    return originWithInteriorOrigins.count(origin);
  }

  /// Return true if any diagnostics have been emitted for interior origins.
  bool hadAnyErrors() const { return interiorOriginErrorEmitted.any(); }

  /// Record that a diagnostic is going to be emitted for the given interior
  /// origin ID. Return true if one has already been emitted.
  bool markErrorEmittedForInteriorOrigin(size_t id) {
    if (interiorOriginErrorEmitted.test(id))
      return true;
    interiorOriginErrorEmitted.set(id);
    return false;
  }

  // Given a mutation of an origin "o", invalidate any interior origins derived
  // from it, like "o[]" or "o.field[]".
  void invalidateOnMutation(TypedAttr mutatedOrigin,
                            TrackedAndInteriorLiveness &liveness,
                            Operation &thisInvalidatingOp) const {
    if (interiorOriginInvalidationMap.empty())
      return;
    processOriginUnionElts(mutatedOrigin, [&](TypedAttr origin) {
      // Clear any interior origins derived from the mutated origin.
      auto it = interiorOriginInvalidationMap.find(origin);
      if (it == interiorOriginInvalidationMap.end())
        return;
      // Zap entries invalided due to the mutation.
      for (ssize_t id = it->second.find_first(); id >= 0;
           id = it->second.find_next(id)) {
        // When we clear it, remember what operation invalidated it.
        auto &entry = liveness.interior[id];
        if (entry.getInt()) {
          entry.setInt(false);
          entry.setPointer(&thisInvalidatingOp);
        }
      }
    });
  }

private:
  /// The function we're analyzing.
  FnOp theFunc;

  /// This method is called whenever we discover an interior origin.
  void addInteriorOrigin(InteriorOriginAttr interior);

  /// Each interior origin we see is assigned a dense unique ID so we can use
  /// bitvector dataflow to track their validity.
  DenseMap<InteriorOriginAttr, size_t> interiorOriginID;

  /// This is the counter for the next assigned ID.
  size_t nextInteriorOriginID = 0;

  /// If there is a subtree origin x.field~ used in this function, this map
  /// notices it and keeps track of the synthetic interior origin for it.  Most
  /// origins won't have a subtree origin: they won't have an entry.
  DenseMap<Attribute, InteriorOriginAttr> subtreeForOriginMap;

  /// This contains entries for origins that have derived interior origins,
  /// indicating what to invalidate when that origin is mutated.
  DenseMap<Attribute, BitVector> interiorOriginInvalidationMap;

  /// This contains entries for origins that have interior origins somewhere
  /// inside of them. This allows us to filter out the majority of origins
  /// references that don't.
  SmallPtrSet<Attribute, 8> originWithInteriorOrigins;

  /// Tracks whether we've already emitted a diagnostic for each interior
  /// origin ID, to avoid redundant errors for the same origin.
  BitVector interiorOriginErrorEmitted;

  void noticeInvalidatedSubtreeOrigins(
      Attribute origin, SmallVector<InteriorOriginAttr, 16> &subTreeInvalidated,
      SmallPtrSet<Attribute, 16> &hasNoSubtreeOrigin);
};
} // namespace

InteriorOriginTracker::InteriorOriginTracker(PerThreadCache &perThreadCache,
                                             FnOp func)
    : theFunc(func) {

  // The same types (eg none) are used over & over again, precache origin scan.
  SmallPtrSet<Type, 32> visitedTypes;
  SmallVector<Type, 32> collectedTypes;

  auto collectType = [&](Type type) {
    if (visitedTypes.insert(type).second)
      collectedTypes.push_back(type);
  };

  // For a function with regions, collect the types from the block arguments.
  auto collectBlockArgs = [&](Operation *op) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region)
        for (auto arg : block.getArguments())
          collectType(arg.getType());
    }
  };

  // Get any interior origins in function arguments (including the result slot).
  collectBlockArgs(func.getOperation());

  // Scan the body region to find all the interior origins.
  func.getBodyRegion().walk([&](Operation *op) -> WalkResult {
    // Skip looking at nested functions, they are handled as separate
    // contexts.
    if (isa<FnOp>(op))
      return WalkResult::skip();

    // Interior origins can be found by scanning all the operation results
    // and arguments.  We can ignore any operands because they'll be using
    // the other things.
    for (Type resultType : op->getResultTypes())
      collectType(resultType);

    // If there are any regions, check the block arguments for arguments.
    collectBlockArgs(op);
    return WalkResult::advance();
  });

  // Given a set of origins used in the function, scan their structure for any
  // interior origins (including ones nested in fields) and build the
  // invalidation map.
  SmallPtrSet<Attribute, 16> allDerivedOrigins;
  for (TypedAttr origin : perThreadCache.originFinder.findOriginsIn(
           collectedTypes, func.getFuncTypeGenerator().getCaptureOrigins())) {
    // processRawOrigin looks through SugarAttr and UnionAttr.
    bool hasInteriorOrigin = false;
    processOriginUnionElts(origin, [&](TypedAttr raw) {
      // Strip off things that might be outside an interior origin, since they
      // won't matter for our analysis.
      while (1) {
        allDerivedOrigins.insert(raw);
        if (auto mutCast = dyn_cast<OriginMutCastAttr>(raw))
          raw = mutCast.getOperand();
        else if (auto field = dyn_cast<OriginFieldAttr>(raw)) {
          raw = field.getBase();
        } else if (auto sugar = dyn_cast<SugarAttr>(raw))
          raw = sugar.getCanonical();
        else
          break;
      }

      // If we find a subtree origin, we will need to generally handle it like
      // and interior origin (e.g. it gets invalidated when the base is
      // mutated, defined as valid when returned from a function), so enumerate
      // it as a synthetic interior origin.
      if (auto subtree = dyn_cast<OriginSubtreeAttr>(raw)) {
        auto synthetic = getInteriorForSubtreeOrigin(subtree);
        subtreeForOriginMap[subtree.getBase()] = synthetic;
        raw = synthetic; // Process this like any other below.
      }

      // If we find an interior origin inside, process it.
      if (auto interior = sugarDynCast<InteriorOriginAttr>(raw)) {
        addInteriorOrigin(interior);
        hasInteriorOrigin = true;
      }
    });
    // Keep track of any origins that have interior origins, so we can rapidly
    // ignore things that don't.
    if (hasInteriorOrigin)
      originWithInteriorOrigins.insert(origin);
  }

  // Now that we found all the subtree origins, make sure they get invalidated
  // when any origins descendent from their bases are mutated. For example:
  //    'a.list~' aill already be listed as being invalidated when 'a' or
  //    'a.list' is mutated, but we need to add 'a.list[other]' and
  //    a.list.field as well.
  if (!subtreeForOriginMap.empty()) {
    SmallPtrSet<Attribute, 16> hasNoSubtreeOrigin;
    SmallVector<InteriorOriginAttr, 16> subTreeInvalidated;
    for (Attribute origin : allDerivedOrigins) {
      noticeInvalidatedSubtreeOrigins(origin, subTreeInvalidated,
                                      hasNoSubtreeOrigin);

      // If there are any subtree origins invalidated by this origin, add them
      // to its invalidation map.
      if (subTreeInvalidated.empty())
        continue;

      auto &bitvector = interiorOriginInvalidationMap[origin];
      for (auto interior : subTreeInvalidated) {
        size_t id = getInteriorOriginID(interior);
        if (id >= bitvector.size())
          bitvector.resize(id + 1);
        bitvector.set(id);
      }
      subTreeInvalidated.clear();
    }
  }

  // Now that we found all the interior origins, make sure all the bitvectors
  // are the same length.
  if (size_t numInteriorOrigins = getNumInteriorOrigins()) {
    for (auto &bitVector : interiorOriginInvalidationMap)
      bitVector.second.resize(numInteriorOrigins);
    interiorOriginErrorEmitted.resize(numInteriorOrigins);
  }
}

/// This method is called whenever we discover a derived origin "X", like
/// "a.list[].first[]" or "a.list.otherfield". It checks to see if there is a
/// subtree origin "X~"
/// invalidated when any of the origins derived from its base are mutated, like
/// "a", "a.list", or "a.list[].first".
void InteriorOriginTracker::noticeInvalidatedSubtreeOrigins(
    Attribute origin, SmallVector<InteriorOriginAttr, 16> &subTreeInvalidated,
    SmallPtrSet<Attribute, 16> &hasNoSubtreeOrigin) {

  // If we've already seen this origin and it has no subtrees, exit early.
  if (hasNoSubtreeOrigin.count(origin))
    return;

  // If this locally has a subtree origin, add it.
  if (auto it = subtreeForOriginMap.find(origin);
      it != subtreeForOriginMap.end())
    subTreeInvalidated.push_back(it->second);

  // If this is a derived origin, recurse.
  TypedAttr base;
  if (auto mutCast = dyn_cast<OriginMutCastAttr>(origin))
    base = mutCast.getOperand();
  else if (auto field = dyn_cast<OriginFieldAttr>(origin))
    base = field.getBase();
  else if (auto sugar = dyn_cast<SugarAttr>(origin))
    base = sugar.getCanonical();
  else if (auto subtree = dyn_cast<OriginSubtreeAttr>(origin))
    base = subtree.getBase();

  if (base)
    noticeInvalidatedSubtreeOrigins(base, subTreeInvalidated,
                                    hasNoSubtreeOrigin);

  // If we didn't find anything, then don't process this again.
  if (subTreeInvalidated.empty())
    hasNoSubtreeOrigin.insert(origin);
}

/// This method is called whenever we discover an interior origin.  This may
/// have internal fields and other interior origins inside.  Enumerate it if it
/// is the first time we've seen it and build the inv+alidation map for any
/// nested origins.
void InteriorOriginTracker::addInteriorOrigin(InteriorOriginAttr interior) {
  // Only scan an interior origin the first time we see it.
  if (!interiorOriginID.insert({interior, nextInteriorOriginID}).second)
    return;
  auto thisID = nextInteriorOriginID;
  ++nextInteriorOriginID;

  // Okay, we've found and numbered this interior origin, that may be rooted
  // off a collection of other field references and interior origins. Process
  // its body.
  TypedAttr body = interior.getBase();

  // If we have a nested interior origin, remember it.
  InteriorOriginAttr nextInterior;
  while (1) {
    // When this attribute is mutated, we need to invalidate the interior
    // origin.
    auto &bitvector = interiorOriginInvalidationMap[body];
    // This will always be the largest ID we've seen so far, since we just
    // allocated it.
    bitvector.resize(thisID + 1);
    bitvector.set(thisID);

    if (auto field = dyn_cast<OriginFieldAttr>(body))
      body = field.getBase();
    else if (auto interior = dyn_cast<InteriorOriginAttr>(body)) {
      if (!nextInterior)
        nextInterior = interior;
      body = interior.getBase();
    } else {
      // Otherwise it must be a leaf. We're done.  Enumerate this to make sure
      // we don't discover any other forms later.
      assert(isa<ParamDeclRefAttr>(body) && "Unknown origin structure");
      break;
    }
  }

  // If there was an interior origin underneath this one, than recursively
  // process it, which will recursively process any other nested ones under it.
  //  ref r2 = a.list[].second.field[]  => a.list[]
  if (nextInterior)
    addInteriorOrigin(nextInterior);
}

void InteriorOriginTracker::printBV(const BitVector &interiorOriginBits,
                                    llvm::raw_ostream &os) {
  os << "{";
  for (ssize_t id = interiorOriginBits.find_first(); id >= 0;
       id = interiorOriginBits.find_next(id))
    os << " #" << id;
  os << " }";
}

void InteriorOriginTracker::dump() const {
  static std::mutex dumpMutex;
  std::lock_guard<std::mutex> lock(dumpMutex);

  auto &os = llvm::errs();
  if (nextInteriorOriginID == 0)
    os << "Empty ";
  os << "InteriorOriginTracker\n";

  if (nextInteriorOriginID == 0)
    return;

  // Invert the map so we can print in order.
  SmallVector<InteriorOriginAttr> invertedMap;
  invertedMap.resize(nextInteriorOriginID);
  for (auto &entry : interiorOriginID)
    invertedMap[entry.second] = entry.first;

  OriginPrinter originPrinter;
  for (auto [idx, entry] : llvm::enumerate(invertedMap)) {
    os << "  #" << idx << " => ";
    originPrinter.print(os, entry, /*elideOriginOf=*/true);
    os << "\n";
  }

  if (!interiorOriginInvalidationMap.empty()) {
    os << "Invalidation map (note: dumped order is unstable):\n";
    for (auto &entry : interiorOriginInvalidationMap) {
      os << "  ";
      originPrinter.print(os, cast<TypedAttr>(entry.first),
                          /*elideOriginOf=*/true);
      os << " => ";
      printBV(entry.second, os);
      os << "\n";
    }
  }
  os << "\n";
}

/// Mark the given interior origin as live in the given bitvector.
void InteriorOriginTracker::markInteriorOriginLive(
    InteriorOriginAttr o, TrackedAndInteriorLiveness &liveness, Operation &op,
    const mlir::DominanceInfo &domInfo) {
  // Interior origins must not have a parametric user name.  We can't do our
  // analysis before elaboration, so don't know what concrete parameter values
  // will be substituted in.  Reject them so the user doesn't misunderstand our
  // capabilities.
  if (!isa<StringAttr>(o.getUserName())) {
    size_t interiorID = getInteriorOriginID(o);
    if (!markErrorEmittedForInteriorOrigin(interiorID))
      op.emitError("interior origin name must be a string literal, it cannot "
                   "be parametric when used");
    return;
  }

  // If the entry was previously invalid, mark it as now live.
  auto &entry = liveness.interior[getInteriorOriginID(o)];
  if (!entry.getInt()) {
    entry.setInt(true);
    entry.setPointer(&op);
    return;
  }

  // Otherwise the entry was already live.  We prefer to keep track of the first
  // definition so the liveness scope of the reference is maximized:
  //    ref a = list[i]
  //    ...
  //    ref b = list[j]
  //    ...
  //    use(a) # This is valid even though "b" also marked it live.
  if (domInfo.dominates(entry.getPointer(), &op))
    return;

  // If the entry was above us but in a nested region, hoist the location up and
  // recheck dominance.  This handles cases where the live reference doesn't SSA
  // dominate the new reference, like:
  //    if cond:
  //       ref a = list[i]
  //       use(a)
  //    else:
  //       abort()
  //    ref b = list[j]   # Here.
  //    use(b)
  //
  // In these cases, we bubble the location of "a" up to the "if", and recheck.
  Operation *liftedDefiningOp = liftToCommonRegion(entry.getPointer(), &op);
  if (domInfo.dominates(liftedDefiningOp, &op))
    return;

  // Otherwise, it isn't dominating, so use it.
  entry.setPointer(&op);
}

//===----------------------------------------------------------------------===//
// UninitializedValueScan
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the second pass over a function body, which
/// identifies and complains about uses of uninitialized values.
///
/// The sentinel slot #0 in the bitvector indicates whether the block is
/// reachable from entry.  It is set to false after terminators so merge points
/// known not to merge the values.
struct UninitializedValueScan {
  UninitializedValueScan(ValueSet &valueSet,
                         InteriorOriginTracker &interiorOriginTracker)
      : valueSet(valueSet), interiorOriginTracker(interiorOriginTracker),
        liveness(valueSet.getNumTotalBits(),
                 interiorOriginTracker.getNumInteriorOrigins()) {}
  UninitializedValueScan(const UninitializedValueScan &existing) = delete;

  void scanFunction(FnOp func);
  void scanBlock(Block &body);

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op);
  void checkElIfOp(HLCF::ElifOp op);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  void diagnoseUsageError(ValueRef valueRef, Operation &op);
  void checkInteriorOriginUsage(InteriorOriginAttr interior, Value operand,
                                Operation &op);
  void checkUse(Value value, Operation &op, bool isDeref);
  void checkDef(Value value, Operation &op, bool isDeref);
  void checkConsume(Value value, Operation &op, bool isDeref);
  void checkMarkDestroyed(Value value, Operation &op);
  void checkOriginAccesses(TypedAttr origin, Value operand, Operation &op);
  void handleAnyOriginUse(Operation &op, ArrayRef<TypedAttr> definedOrigins);

  SmallVector<InteriorOriginAttr> findInteriorOriginsInType(Type type);
  void invalidateInteriorOriginsForOperand(Value operand, bool isDeref,
                                           Operation &op);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is information about all the interior origins used in the function.
  InteriorOriginTracker &interiorOriginTracker;

  /// This is the set of values and interior origins known to be live at this
  /// point.
  TrackedAndInteriorLiveness liveness;

  /// When analyzing the body of a loop, this bitset indicates what a 'continue'
  /// should intersect with.
  TrackedAndInteriorLiveness *continueSet = nullptr;
  /// When analyzing the body of a loop, this bitset indicates what a 'break'
  /// should intersect with.
  TrackedAndInteriorLiveness *breakSet = nullptr;
  /// When analyzing the body of a try, this stack indicates what a
  /// 'raise' should intersect with. It is indexed with a raise label.
  RaiseSetEntry<TrackedAndInteriorLiveness> *raiseEntryInfo = nullptr;
};
} // namespace

[[maybe_unused]] void UninitializedValueScan::dump() const {
  auto &os = llvm::errs();
  if (valueSet.getValueInfos().size() < 10) {
    valueSet.dump();
    os << "\n";
  }

  os << "UninitializedValueScan for ";
  valueSet.printFuncName(os);
  os << "\n  live = ";
  liveness.print(valueSet, os);
  os << "\n";

  if (RaiseSetEntry<TrackedAndInteriorLiveness> *curr = raiseEntryInfo) {
    os << "  raise: {";
    while (curr) {
      os << curr->label << " : ";
      curr->raiseSet->print(valueSet, os) << "\n";
      curr = curr->prev;
    }
    os << " }\n";
  }

  if (breakSet) {
    os << "  break: ";
    breakSet->print(valueSet, os) << "\n";
  }
  if (continueSet) {
    os << "  continue: ";
    continueSet->print(valueSet, os) << "\n";
  }
  interiorOriginTracker.dump();

  os.flush();
}

static Type digIntoTypeAtFieldOffset(Type type, unsigned firstInvalidOffset,
                                     unsigned nextValidOffset,
                                     InFlightDiagnostic &diag,
                                     TypeDeclInfo &typeDeclInfo) {
  // Dig into the type to get to the right field.
  while (firstInvalidOffset) {
    // If this is the full-object bit for this entire type, then we found the
    // problem.
    if (firstInvalidOffset + 1 == typeDeclInfo.getNumFieldsInType(type))
      return type;

    // To index into this type, it must be a DeclRef.
    auto declRefType = sugarCast<LIT::StructType>(type);

    auto [fieldDecl, fieldBitOffset, numFieldBits] =
        typeDeclInfo.getFieldContaining(declRefType, firstInvalidOffset);
    firstInvalidOffset -= fieldBitOffset;
    nextValidOffset -= fieldBitOffset;
    type = fieldDecl.getReboundType(declRefType);
    diag << "." << fieldDecl.getName();
  }

  // Dig into the field to ignore trailing members that we don't care about.
  while (nextValidOffset < typeDeclInfo.getNumFieldsInType(type)) {
    auto declRefType = sugarCast<LIT::StructType>(type);
    auto [fieldDecl, startBit, numBits] =
        typeDeclInfo.getFieldContaining(declRefType, 0);
    type = fieldDecl.getReboundType(declRefType);
    diag << "." << fieldDecl.getName();
  }

  return type;
}

/// When complaining about a specific value, check to see if the /entire/
/// field-sensitive value is missing from the specified bitvector.  If not,
/// add a suffix that identifies the first whole field that is missing.
static void addBadValueNameToDiag(ValueRef valueRef, const BitVector &bits,
                                  ValueSet &valueSet,
                                  mlir::InFlightDiagnostic &diag) {
  const ValueInfo &valueEntry = valueSet.getValueInfo(valueRef.valueId);

  diag << "'" << valueEntry.getName().str();
  // If the whole value is missing, then don't add any field information.
  if (valueEntry.getFullValueRef(valueRef.valueId).isAllMissing(bits)) {
    diag << "'";
    return;
  }

  // Figure out what the end of the field bits are so we can report the first
  // fields.  The full object ends with a bit to track whether the whole value
  // is initialized which we don't want to track.
  unsigned fullValueStartBit = valueEntry.startValueBit;

  unsigned endOfFullObjectFields = valueEntry.endValueBit - 1;
  if (endOfFullObjectFields == fullValueStartBit) {
    // No stored fields!
    diag << "'";
    return;
  }

  // The end of the reference is either the end of valueref (if that was a
  // subfield of the overall object) or it is the end of full object.
  unsigned endOfAccessFields = std::min(endOfFullObjectFields, valueRef.endBit);

  // We know that something in valueRef is missing, but we don't know which
  // piece.  Find the first bit in valueRef that isn't live.
  unsigned firstMissingFieldNo =
      std::min(unsigned(bits.find_next_unset(valueRef.startBit - 1U)),
               endOfAccessFields - 1);
  // Find the area of overlap so we complain about larger aggregates that are
  // fully uninit, not tiny parts of them.
  unsigned firstPresentFieldNo = std::min(
      unsigned(bits.find_next(firstMissingFieldNo)), endOfAccessFields);

  // Ok, the uninitialized thing is [firstMissingFieldNo, firstPresentFieldNo)
  // so we want to figure out which sub-piece of the whole value type is the
  // problem, and identify a path that drills down through each of the named
  // fields.
  auto type = valueRef.getValueType(valueEntry.value);
  // Emit the field prefix for the specified type.
  digIntoTypeAtFieldOffset(type, firstMissingFieldNo - fullValueStartBit,
                           firstPresentFieldNo - fullValueStartBit, diag,
                           valueSet.getTypeDeclInfo());
  diag << "'";
}

SmallVector<InteriorOriginAttr>
UninitializedValueScan::findInteriorOriginsInType(Type type) {
  SmallVector<InteriorOriginAttr> result;

  // Don't scan anything if no interior origins are accessed.
  if (!liveness.interior.empty()) {
    for (auto origin :
         valueSet.perThreadCache.originFinder.findOriginsIn(type, {})) {
      // Ignore the vast majority of origins that don't have interior origins.
      if (!interiorOriginTracker.originHasInteriorOrigins(origin))
        continue;

      // Given a def of something with an "a.list["x"].second.field["y"].z"
      // origin, we need to mark the "x" and "y" interior origins live.
      // Subtree origins (~a) also need the same handling.
      origin.walk([&](Attribute nested) {
        if (auto interior = dyn_cast<InteriorOriginAttr>(nested)) {
          result.push_back(interior);
          return;
        }
        // Subtree origins may have erased an interior origin.  Add a
        // placeholder for it.
        if (auto subtree = dyn_cast<OriginSubtreeAttr>(nested)) {
          result.push_back(getInteriorForSubtreeOrigin(subtree));
          return;
        }
      });
    }
  }
  return result;
}

void UninitializedValueScan::invalidateInteriorOriginsForOperand(
    Value operand, bool isDeref, Operation &op) {
  // Ignore the results of function calls etc.  These are defining the result,
  // not mutating an already valid reference.
  if (!isDeref)
    return;

  // Given a mutation of an origin "o", invalidate any interior origins derived
  // from it, like "o[]" or "o.field[]".
  interiorOriginTracker.invalidateOnMutation(
      cast<RefType>(operand.getType()).getOrigin(), liveness, op);
}

/// Verify that the specified ValueRef is live at this point, diagnosing an
/// error at the specified operation if not.
void UninitializedValueScan::checkUse(Value value, Operation &op,
                                      bool isDeref) {
  SmallVector<InteriorOriginAttr> interiorOrigins;
  SmallVector<ValueRef> accesses;

  // If this is a direct reference to a value, return field sensitive info.
  if (ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref)) {
    accesses.push_back(valueRef);

    auto valueType = ValueRef::getDereferencedType(value.getType(), isDeref);
    interiorOrigins = findInteriorOriginsInType(valueType);
  } else {
    accesses = valueSet.getValueRefsForAccess(value, isDeref, interiorOrigins);
  }

  for (ValueRef access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveness.tracked))
      diagnoseUsageError(access, op);
  }

  // Verify that any accessed interior origins are also live.
  for (auto origin : interiorOrigins)
    checkInteriorOriginUsage(origin, value, op);
}

/// One of the specified fields is missing, so emit an error about it.  This is
/// to complain about incorrect accesses to an value.
void UninitializedValueScan::diagnoseUsageError(ValueRef valueRef,
                                                Operation &op) {
  // Ok, it isn't, gear up to see how to best report the error.
  ValueInfo &valueInfo = valueSet.getValueInfo(valueRef.valueId);

  // As a very unprincipled hack, allow uninitialized values at the end of REPL
  // cells. The reason we need this is that the REPL "persists" values onto the
  // heap with an IR rewrite, and does so before lifetime checking.  As such,
  // it has no idea what values are live out of the end of each cell.  This
  // needs to be fixed, but allow simple things like integers to "work" for now.
  // This will not work at all for non-trivial values though because
  // reassignments over them will assume they are initialized.
  if (valueInfo.value.getDefiningOp<LIT::RefFromPointerREPLOp>())
    return;

  auto diagOr = valueInfo.emitErrorIfNotDiagnosed(op.getLoc());
  if (!diagOr)
    return;
  auto &diag = *diagOr;

  // If the fields are all valid except for the whole-object bit, then the user
  // tried to initialize a value by initializing all its fields.  Reject this
  // with a customized error.
  std::string valueName = valueInfo.getName().str();
  if (valueRef.isIndirect && valueRef.endBit == valueInfo.endValueBit &&
      valueRef.getSubfield(0, valueRef.getNumBits() - 1)
          .isAllPresent(liveness.tracked) &&
      valueRef.getNumBits() != 1) {
    diag << "'" << valueName
         << "' used with all fields manually initialized "
            "but without calling an '__init__' method";
    diag.attachNote(valueInfo.value.getLoc())
        << "'" << valueName << "' declared here";
    return;
  }

  // Specialize diagnostics for returns because it can be confusing why they are
  // "using" argument values otherwise.
  if (isa<KGEN::ReturnOp>(op)) {
    addBadValueNameToDiag(valueRef, liveness.tracked, valueSet, diag);
    diag << " is uninitialized at ";

    // Diagnostics with implicit function returns can be confusing because the
    // Location of the return op is set to the function entry.  Make it
    // explicit when we're complaining about this.
    if (op.getLoc() == valueSet.getFuncLocation())
      diag << "the implicit ";

    diag << "return from this function";
  } else {
    diag << "use of uninitialized value ";

    // If some fields are present and others are missing, complain about the
    // first whole field that is missing.
    addBadValueNameToDiag(valueRef, liveness.tracked, valueSet, diag);
  }
  diag.attachNote(valueInfo.value.getLoc())
      << "'" << valueName << "' declared here";
}

void UninitializedValueScan::checkDef(Value value, Operation &op,
                                      bool isDeref) {
  // Invalidate any interior origins derived from the defined value.
  invalidateInteriorOriginsForOperand(value, isDeref, op);

  // Direct accesses are handled in a field sensitive way, and this can count as
  // an initialization.
  if (ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref)) {
    // Finally, marks its value live so any use after this isn't treated as
    // uninitialized.
    valueRef.markBits(liveness.tracked, true);
    return;
  }

  // If this is an indirect reference then a mutation will require that all
  // values being mutated are initialized, because we cannot perform field
  // sensitive initialization, only overwrite/mutate.
  SmallVector<InteriorOriginAttr> interiorOrigins;
  SmallVector<ValueRef> accesses =
      valueSet.getValueRefsForAccess(value, isDeref, interiorOrigins);
  for (auto access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveness.tracked))
      diagnoseUsageError(access, op);
  }

  // Mark any defined interior origins as live.
  for (auto origin : interiorOrigins)
    interiorOriginTracker.markInteriorOriginLive(origin, liveness, op,
                                                 valueSet.domInfo);
}

void UninitializedValueScan::checkConsume(Value value, Operation &op,
                                          bool isDeref) {
  ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref);
  if (!valueRef) {
    // We cannot consume an indirect value (unless it is untracked).
    if (!valueSet.isTrivial(value, isDeref) &&
        // FIXME(#29005): AnyRefType binds to non-trivial types
        isDeref) {
      ValueInfo &valueInfo = valueSet.getValueInfo(valueRef.valueId);
      valueInfo.emitErrorIfNotDiagnosed(
          op.getLoc(), "cannot consume indirect references to values");
    }

    // Calls and returns that consuming an RP value can access all the interior
    // origins inside the RP value. Collect and check them.
    auto valueType = ValueRef::getDereferencedType(value.getType(), isDeref);
    for (auto origin : findInteriorOriginsInType(valueType))
      checkInteriorOriginUsage(origin, value, op);
    return;
  }

  // The value must be completely live in order for us to consume it.  If not,
  // use "checkUse" to diagnose the problem.
  if (!valueRef.isAllPresent(liveness.tracked))
    diagnoseUsageError(valueRef, op);

  // If tracked, marks its value as dead.
  valueRef.markBits(liveness.tracked, false);

  // Invalidate any interior origins derived from the consumed value.
  invalidateInteriorOriginsForOperand(value, isDeref, op);
}

/// The lit.ownership.mark_destroyed op consumes the whole object bit of
/// a value only, but not its fields.  It marks the final aggregate as
/// uninitialized.
void UninitializedValueScan::checkMarkDestroyed(Value value, Operation &op) {
  ValueRef access = valueSet.getDirectValueRef(value, /*isDeref=*/true);
  ValueInfo &valueInfo = valueSet.getValueInfo(access.valueId);
  if (!access) {
    valueInfo.emitErrorIfNotDiagnosed(
        op.getLoc(), "can only mark directly tracked values as destroyed");
    return;
  }
  // This operation is only generated on 'self' of deinit, so this should never
  // trigger, but make sure only entire values are destroyed.
  if (access != valueInfo.getFullValueRef(access.valueId)) {
    valueInfo.emitErrorIfNotDiagnosed(
        op.getLoc(), "can only mark full values as destroyed, not subfields");
    return;
  }

  // Ignore a mark_destroyed if the whole value is already destroyed. This can
  // happen when a deinit method transfers self to another deinit method.
  if (access.isAllMissing(liveness.tracked)) {
    op.setAttr(unusedMarkDestroyName, UnitAttr::get(op.getContext()));
    return;
  }

  // Check that the consumed bit is live, otherwise it cannot be destroyed.
  ValueRef fullObjectBit = access.getSubfield(access.getNumBits() - 1, 1);

  // If not, then there is an error which we diagnose.
  if (!fullObjectBit.isAllPresent(liveness.tracked))
    diagnoseUsageError(fullObjectBit, op);

  // From this point on, the whole value is uninitialized.
  access.markBits(liveness.tracked, false);

  // Invalidate any interior origins derived from the consumed value.
  invalidateInteriorOriginsForOperand(value, /*isDeref=*/true, op);
}

/// Given an operand that uses an interior origin, figure out what program point
/// defined it.  For example, the operand might be a !lit.ref for a ref operand,
/// or a struct value like a Pointer that contains the origin, or it might be
/// something like a by-ref DictIterator that contains the origin.
static Operation *getProgramPointThatDefinedInteriorOrigin(Value v) {
  // FIXME: This doesn't seem enough to handle the by-ref operand case, it will
  // get the reference to the VarDecl, not necessarily the store.  This is
  // "close enough" for now, but should be evaluated as things shape up.
  while (1) {
    // Look through stuff that is generally transparent to CheckLifetimes.
    // Note that this doesn't look through RefUpcastOp, since it can define
    // interior origins.
    if (auto structGER = v.getDefiningOp<RefStructGEROp>())
      v = structGER.getOperand();
    else if (auto rebind = v.getDefiningOp<RebindOp>())
      v = rebind.getOperand();
    else if (auto immut = v.getDefiningOp<RefImmutOp>())
      v = immut.getOperand();
    else if (auto load = v.getDefiningOp<RefLoadOp>()) {
      // Looking through ref loads will give us the vardecl for local values,
      // which is strictly more conservative than looking at the load itself:
      //
      //   %vd = lit.var.decl
      //   ... some stuff ...
      //   %tmp = lit.ref.load %vd
      //
      // Here we treat the program point for the reference as %vd instead of
      // %tmp.
      v = load.getOperand();
    } else if (auto load = v.getDefiningOp<LoadConsumeOp>()) {
      v = load.getOperand(); // Same as LoadOp.
    } else if (auto varDecl = v.getDefiningOp<VarDeclOp>()) {
      // If we have a vardecl, scan down its block to find the first use that
      // might be a store to it.  We know the vardecl itself doesn't have an
      // interior origin as its origin, so this must be something nested within
      // it (e.g. a Pointer or Span).  We scan to avoid problems like:
      //    %vd = lit.var.decl
      //    ...
      //    ref r1 = get_interior_origin()
      //    store r1 -> %vd
      //    ...
      //    use(%vd)
      // If we didn't do this, we'd consider the access to start at the var.decl
      // which is earlier than the interior origin is alive from. Instead we
      // want to consider the vardecl live at the first store.  This is very
      // conservative correct, but could be made more aggressive if needbe.
      auto *block = varDecl->getBlock();
      for (Block::iterator it = varDecl->getIterator(); it != block->end();
           ++it) {
        if (llvm::any_of(it->getOperands(),
                         [&](Value operand) { return operand == v; }))
          return &*it;
      }
      // If we found no uses of the vardecl, we could return the terminator, but
      // be more conservative than that - return the VD itself.
      return varDecl;
    } else if (Operation *definingOp = v.getDefiningOp()) {
      return definingOp;
    } else {
      // A block argument has no defining operation, but it is available from
      // the moment its region is entered, so the enclosing operation is its
      // program point. This is how an interior origin reaches a synthesized
      // closure-storage initializer: the origin is carried by the type of a
      // function parameter, so it is live on the function itself.
      return cast<BlockArgument>(v).getOwner()->getParentOp();
    }
  }
}

static std::string getOriginStr(InteriorOriginAttr origin) {
  std::string originStr;
  llvm::raw_string_ostream originOs(originStr);
  OriginPrinter().print(originOs, origin, /*elideOriginOf=*/true);
  return originStr;
}

void UninitializedValueScan::checkInteriorOriginUsage(
    InteriorOriginAttr interior, Value operand, Operation &op) {

  // If we know it isn't live because it has never been invalidated, or got
  // invalidated, then it is obviously not live.
  size_t interiorID = interiorOriginTracker.getInteriorOriginID(interior);
  auto &entry = liveness.interior[interiorID];
  if (!entry.getInt()) {
    // If this is being defined by a lit.ref.from_pointer, then this is the
    // internal implementation of _get_ref_with_unsafe_interior_origin. Allow
    // use of the interior origin.
    if (isa<KGEN::ReturnOp>(op) &&
        RefImmutOp::stripRebinds(operand).getDefiningOp<RefFromPointerOp>())
      return;

    // If the interior origin is based on something else uninitialized, then
    // complain about the base of the access, not its internals.
    SmallVector<InteriorOriginAttr> dummy;
    for (ValueRef access :
         valueSet.getValueRefsForOrigin(interior.getBase(), dummy)) {
      if (!access.isAllPresent(liveness.tracked)) {
        diagnoseUsageError(access, op);
        return;
      }
    }

    // If we already emitted a diagnostic for this, don't do it again.
    if (interiorOriginTracker.markErrorEmittedForInteriorOrigin(interiorID))
      return;

    auto diag = emitError(op.getLoc());
    if (Operation *invalidatingOp = entry.getPointer()) {
      diag << "use of invalidated interior reference '"
           << getOriginStr(interior) << "'";
      diag.attachNote(invalidatingOp->getLoc())
          << "origin was invalidated here";
    } else {
      diag << "use of a never-initialized interior reference '"
           << getOriginStr(interior) << "'";
    }
    return;
  }

  // If the interior origin is available at this program point, we still need
  // to verify that the actual reference being used hasn't been invalidated
  // and reinstated.  Consider something like:
  //
  //     ref r1 = list[i] # Def #1
  //     list.append(4) # Could reallocate the memory 'r1' points to.
  //     ref r2 = list[j] # Def #2
  //     use(r2)  # This is clearly fine.
  //     use(r)   # This is invalid because 'r' was invalidated.
  //
  // To handle this, we look at the operation in the interior origin entry,
  // which is the first operation that defined the interior origin.  If that
  // operation dominates the reference, then it is valid (as in the case of
  // Def #2 dominating r2). If that operation does not dominate the reference
  // (as in Def #2 not dominating r1) then it is invalid.
  Operation *definingOp = entry.getPointer();
  Operation *operandPt = &op;

  // If this is checking liveness of a result slot for a return, use the
  // location of the return itself. "operand" may be null when checking capture
  // set origins.
  if (operand && !isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op))
    operandPt = getProgramPointThatDefinedInteriorOrigin(operand);

  assert(operandPt && "operand wasn't defined by a program point?");

  // If the defining operation dominates the operand, then it is valid.
  if (valueSet.domInfo.dominates(definingOp, operandPt))
    return;

  // Otherwise, we may have a more complex case where the defining op is buried
  // in some control structure (e.g. a "then" block where the "else" block
  // doesn't fall through:
  //      var ptr
  //      if cond:
  //        ptr = foo() # defining op.
  //      else:
  //        return
  //      use(ptr)
  // Handle this by climbing up the region tree until \p definingOp and
  // \p operandPt share a region, then checking dominance there.
  Operation *liftedDefiningOp = liftToCommonRegion(definingOp, &op);
  Operation *liftedOperandPt = liftToCommonRegion(operandPt, &op);
  if (valueSet.domInfo.dominates(liftedDefiningOp, liftedOperandPt))
    return;

  // TODO(remove legacy closures): Legacy closures can have operands defined
  // outside the function that are referenced directly from within it.  For
  // example:
  //   ptr = ...
  //   @__parameter
  //   def inner():
  //     use(ptr)
  // This gets represented as a direct use of the operand but we mark the origin
  // live on the function itself. This trips up the dominance check, so special
  // case this.
  if (isa<FnOp>(definingOp))
    return;

  // Otherwise, it is valid for some other reference but was reinitialized since
  // this reference was formed.
  if (interiorOriginTracker.markErrorEmittedForInteriorOrigin(interiorID))
    return;

  auto diag = emitError(op.getLoc(), "use of invalidated interior reference '");
  diag << getOriginStr(interior) << "'";
  // Control-flow parents (e.g. the `if` at a branch join) can be the merged
  // defining point without being a concrete redefinition site.
  if (definingOp->getNumRegions() == 0)
    diag.attachNote(definingOp->getLoc())
        << "origin was defined here, after the reference was formed";
  else
    diag.attachNote(definingOp->getLoc())
        << "origin was defined inside this control structure";
}

/// Check any unstructured origins that are accessed by the operation.
void UninitializedValueScan::checkOriginAccesses(TypedAttr origin,
                                                 Value operand, Operation &op) {
  SmallVector<InteriorOriginAttr> interiorOrigins;
  SmallVector<ValueRef> accesses =
      valueSet.getValueRefsForOrigin(origin, interiorOrigins);
  for (auto access : accesses) {
    // The referenced value fields must be live.
    if (!access.isAllPresent(liveness.tracked))
      diagnoseUsageError(access, op);
  }

  // Verify that any accessed interior origins are also live.
  for (auto origin : interiorOrigins)
    checkInteriorOriginUsage(origin, operand, op);
}

/// This function is called when an operation uses a #lit.any.origin origin.
/// This happens when the operation accesses through (e.g.) an unbound
/// UnsafePointer.  We don't know what objects may be touched by this access,
/// but we want to ensure (for usability sake) that any origin-tracked values
/// are treated as a use, so they don't get destroyed too early.
///
/// We handle this by learning which things need extension in this function,
/// then attaching an attribute that destructor insertion pass will notice in
/// the second pass.
void UninitializedValueScan::handleAnyOriginUse(
    Operation &op, ArrayRef<TypedAttr> definedOrigins) {
  // Turn the list of origins (which might include unions, mutcasts, etc) into
  // the raw underlying origins of values.
  SmallPtrSet<Attribute, 8> definedOriginSet;
  for (auto elt : definedOrigins) {
    // Look through imm cast and unions to find the underlying attrs.
    processOriginUnionElts(getCanonicalAttr(elt), [&](TypedAttr raw) {
      // Ignore field sensitivity of the use: if we have a def of a subfield of
      // the value then we treat it as defining the value.
      while (auto field = dyn_cast<OriginFieldAttr>(raw))
        raw = field.getBase();
      definedOriginSet.insert(raw);
    });
  }

  // Collect a set of value ID's that might be accessed, evaluating each one.
  SmallVector<int32_t> valueIdsToExtend;

  for (unsigned i = 0, e = valueSet.getValueInfos().size(); i != e; ++i) {
    auto &valueInfo = valueSet.getValueInfo(i);
    // Don't mess with things that are in SSA registers - they aren't
    // addressable with a origin.
    if (!valueInfo.value || !valueInfo.isIndirect)
      continue;

    // Can't be a use if the value isn't fully alive here.
    if (!valueSet.getFullValueRef(i).isAllPresent(liveness.tracked))
      continue;

    // Check to see if the operation directly initializes this origin
    // (e.g. by initializing it). If so, we don't want to treat this as a
    // generalized use.
    auto valueOrigin = cast<RefType>(valueInfo.value.getType()).getOrigin();
    if (definedOriginSet.count(getCanonicalAttr(valueOrigin)))
      continue;

    // Check to see if the value is dominated by this op.  It is possible for
    // values to be fully live that are not reachable, e.g.:
    //
    //     if cond:
    //        var thing = ...
    //        use(thing)
    //     else:
    //        return
    //     # Thing is fully initialized here but doesn't dominate.
    //     use_any_origin(..)
    //
    if (!valueSet.domInfo.properlyDominates(valueInfo.value, &op))
      continue;

    // This value might be accessed, so we want to extend its origin if
    // necessary.
    valueIdsToExtend.push_back(i);
  }

  if (valueIdsToExtend.empty())
    return;
  op.setAttr(extraOriginUsesAttrName,
             mlir::DenseI32ArrayAttr::get(op.getContext(), valueIdsToExtend));
}

void UninitializedValueScan::scanFunction(FnOp func) {
  // Initialize the 'liveness' with all the elements that are live-in. The
  // sentinel slot #0 is treated by OriginTrackable as live-in and dead-out
  // which naturally works with our terminators.  The bits are already allocated
  // by our constructor.
  for (const ValueInfo &info : valueSet.getValueInfos()) {
    if (!info.startsUninit) {
      // If the whole value is live on entry, notice that.
      liveness.tracked.set(info.startValueBit, info.endValueBit);
    } else if (info.isFullObjectLiveOnEntry) {
      // If /just/ the full object bit is live on entry, set it.
      liveness.tracked.set(info.endValueBit - 1);
    }
  }

  // Function arguments can have interior origins defined in them (the caller
  // will validate that they are live), so make sure to notice them and set them
  // as live on entry. Similarly, closures can access interior origins from
  // their capture set without defining them in the body.
  SmallVector<InteriorOriginAttr> liveInInteriorOrigins;
  if (!liveness.interior.empty()) {
    SmallVector<Type> argTypes;
    for (auto [arg, conv] :
         llvm::zip(func.getArguments(),
                   func.getFuncTypeGenerator().getArgConventions())) {
      if (!isResultSlot(conv))
        argTypes.push_back(arg.getType());
    }

    // Find all the interior origins in the arguments + capture set.
    for (TypedAttr origin : valueSet.getOriginFinder().findOriginsIn(
             argTypes, func.getFuncTypeGenerator().getCaptureOrigins())) {
      origin.walk([&](Attribute nested) {
        if (auto into = dyn_cast<InteriorOriginAttr>(nested)) {
          liveInInteriorOrigins.push_back(into);
        } else if (auto subtree = dyn_cast<OriginSubtreeAttr>(nested))
          liveInInteriorOrigins.push_back(getInteriorForSubtreeOrigin(subtree));
      });
    }

    // Mark all the capture set origins live on entry.
    for (auto interior : liveInInteriorOrigins)
      interiorOriginTracker.markInteriorOriginLive(
          interior, liveness, *func.getOperation(), valueSet.domInfo);
  }

  // Scan the body of the function.
  scanBlock(func.getBodyRegion().front());

  // The function must not have invalidated any live-in interior origins.
  for (auto interior : liveInInteriorOrigins) {
    size_t interiorID = interiorOriginTracker.getInteriorOriginID(interior);
    Operation *invalidatingOp = liveness.interior[interiorID].getPointer();
    if (invalidatingOp == func.getOperation())
      continue;

    // If we already emitted a diagnostic for this, don't do it again.
    if (interiorOriginTracker.markErrorEmittedForInteriorOrigin(interiorID))
      continue;

    emitError(invalidatingOp->getLoc())
        << "incorrect invalidation of interior origin in closure '"
        << getOriginStr(interior) << "'";
  }
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void UninitializedValueScan::scanBlock(Block &block) {
  OperationEffects opEffects(valueSet.getOriginFinder());
  SmallVector<TypedAttr> definedOrigins;
  for (Operation &op : block) {
    definedOrigins.clear();
    auto overall = opEffects.analyze(op);
    /// If the operation is unknown, ignore it.
    if (overall == OverallOpValueEffect::unknownOp) {
      // NOTE: Can log here when extending things.
      // op.dump();
      continue;
    }

    bool hasAnyOrigin = false;

    // Handle all the normal operand and result effects.
    for (auto [operand, effect] : opEffects.operands) {
      switch (effect) {
      case OperandEffect::regUse:
        checkUse(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::regConsume:
        checkConsume(operand, op, /*isDeref=*/false);
        break;
      case OperandEffect::memLoad:
        hasAnyOrigin |= sugarIsa<AnyOriginAttr>(
            cast<RefType>(operand.getType()).getOrigin());
        checkUse(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memStoreOwned:
        hasAnyOrigin |= sugarIsa<AnyOriginAttr>(
            cast<RefType>(operand.getType()).getOrigin());
        checkDef(operand, op, /*isDeref=*/true);
        definedOrigins.push_back(cast<RefType>(operand.getType()).getOrigin());
        break;
      case OperandEffect::memMut:
        hasAnyOrigin |= sugarIsa<AnyOriginAttr>(
            cast<RefType>(operand.getType()).getOrigin());
        checkUse(operand, op, /*isDeref=*/true);
        checkDef(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memConsume:
        hasAnyOrigin |= sugarIsa<AnyOriginAttr>(
            cast<RefType>(operand.getType()).getOrigin());
        checkConsume(operand, op, /*isDeref=*/true);
        break;
      case OperandEffect::memMarkDestroyed:
        // Mark destroyed doesn't do general origin access.
        checkMarkDestroyed(operand, op);
        break;
      }
    }

    // Process any origins accessed indirectly.
    for (auto [origin, value] : opEffects.origins) {
      checkOriginAccesses(origin, value, op);
      hasAnyOrigin |= sugarIsa<AnyOriginAttr>(origin);
    }

    assert(opEffects.results.size() == op.getNumResults() &&
           "OperationEffects::analyze returned wrong # effects");
    for (auto [result, effect] :
         llvm::zip(op.getResults(), opEffects.results)) {
#ifndef NDEBUG
      OriginTrackable trackable(result);
      // Perform some general sanity checking of the OriginTrackable
      // implementation.

      // Since this is an op result, the live in/out behavior must match each
      // other: if this weren't true, then control flow paths that didn't cross
      // the op could never be satisfied.
      bool endsUninit = false;
      if (trackable) {
        assert((trackable.endInitState == OriginTrackable::EndsInit ||
                trackable.endInitState == OriginTrackable::EndsUninit) &&
               "invalid end init state for an op result");
        endsUninit = trackable.endInitState == OriginTrackable::EndsUninit;
        assert(trackable.startsUninit == endsUninit &&
               "op results must have same live in/out behavior");
      }
#endif

      switch (effect) {
      case ResultEffect::ignore:
        assert(!trackable && "Origin trackable and CheckLifetimes disagree");
        continue;
      case ResultEffect::regDefine:
        assert(trackable && !trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        checkDef(result, op, /*isDeref=*/false);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true);
        break;
      case ResultEffect::memDefineUninitToUninit:
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        assert(trackable && trackable.isIndirect && !endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToUninit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        assert(trackable && trackable.isIndirect && endsUninit &&
               "Origin trackable and CheckLifetimes disagree");
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true);
        definedOrigins.push_back(cast<RefType>(result.getType()).getOrigin());
        break;
      }
    }

    // If the operation used a #lit.any.origin value, then we treat it as an
    // implicit use of all tracked values.  This ensures that the values are
    // not destroyed too early.
    if (hasAnyOrigin)
      handleAnyOriginUse(op, definedOrigins);

    // Mark any interior origins declared by this operation as live.
    for (auto interiorOrigin : opEffects.interiorOriginsDefined)
      interiorOriginTracker.markInteriorOriginLive(interiorOrigin, liveness, op,
                                                   valueSet.domInfo);

    // Finally, handle any other special per-operation behavior.
    switch (overall) {
    case OverallOpValueEffect::unknownOp:
    case OverallOpValueEffect::allHandled:
      // No special action.
      break;
    case OverallOpValueEffect::terminatorOp:
      checkTerminatorOp(op);
      break;
    case OverallOpValueEffect::localControlFlowOp:
      checkLocalControlFlowOp(op);
      break;
    case OverallOpValueEffect::ifLikeOp:
      checkIfLikeOp(op);
      break;
    case OverallOpValueEffect::elifOp: {
      auto elifOp = cast<HLCF::ElifOp>(op);
      // A single if/else shaped elif has the same region layout as IfOp.
      if (elifOp.getElifRegions().empty())
        checkIfLikeOp(op);
      else
        checkElIfOp(elifOp);
      break;
    }
    case OverallOpValueEffect::loopOp:
      checkLoopOp(op);
      break;
    case OverallOpValueEffect::tryOp:
      checkTryOp(cast<LIT::TryOp>(op));
      break;
    }
  }
}

/// Return true if the value is uninitialized at the given exit from the
/// function. A value may be always uninitialized or initialized, or it may be
/// depending on the exit kind.
static bool isUninitializedAtExit(const ValueInfo &valueInfo, Operation &exit) {
  if (valueInfo.endInitState == OriginTrackable::EndsUninit)
    return true;

  if (valueInfo.endInitState == OriginTrackable::InitOnNormal)
    return isa<ErrorReturnOp>(exit);

  if (valueInfo.endInitState == OriginTrackable::InitOnError)
    return isa<KGEN::ReturnOp>(exit);
  return false;
}

/// This is called when the op is a return, lit.error_return or unreachable op.
void UninitializedValueScan::checkTerminatorOp(Operation &op) {
  // If this is a kgen.return then we have an exit from the function
  // (including early returns and exception raises that leave the function).
  // Check that *all* of the values are live-out of the function are
  // initialized.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op)) {
    for (const ValueInfo &valueInfo :
         llvm::drop_begin(valueSet.getValueInfos())) {
      // If the value doesn't need to be live at end of function, ignore it.
      if (isUninitializedAtExit(valueInfo, op)) {
        // For any throw from an initializer, record when self is not fully
        // initialized on this error return (so downstream can avoid demanding
        // destruction of self as a whole).
        //
        // If all the fields are initialized at this point then 'self' is fully
        // initialized so we don't do partial destruction.  If nothing in the
        // object is live then something consumed the full object bit, so we
        // also don't want partial destruction.
        if (isa<LIT::ErrorReturnOp>(op) && valueInfo.isFullObjectLiveOnEntry &&
            !valueInfo.getFullValueRef(0).isAllMissing(liveness.tracked) &&
            !valueInfo.getFullValueRef(0).isAllPresent(liveness.tracked))
          op.setAttr(selfPartiallyInitializedAttrName,
                     UnitAttr::get(op.getContext()));
        continue;
      }

      // If this is the hacky RefFromPointerREPLOp op (used by the REPL
      // only!) and if this is an error path, then we look the other way at
      // indiscretions.
      if (valueInfo.value.getDefiningOp<RefFromPointerREPLOp>() &&
          isa<LIT::ErrorReturnOp>(op))
        continue;

      // Otherwise, it must be live at return/raise.
      checkUse(valueInfo.value, op, /*isDeref=*/valueInfo.isIndirect);
    }
  } else {
    auto unreachable = cast<KGEN::UnreachableOp>(op);

    // Calls to no-return functions (like abort(), exit()) are treated
    // specially: after the call, we allow any live values to be outstanding,
    // because the code isn't reachable. For example, consider:
    //    def f(mut x: Int, mut y: Int):
    //      take(x^) # x is uninit here.
    //      if cond:
    //        abort()
    //      x = "restored"
    // At the abort() call, x is uninitialized but y is initialized.  This is
    // fine, but we need dtor analysis to know what set of values to demand.  Do
    // so by recording what values IDs are fully alive.
    if (unreachable.getIsAfterUnreachableCall()) {
      SmallVector<int32_t> liveValueIds;
      for (unsigned i = 0, e = valueSet.getValueInfos().size(); i != e; ++i) {
        const auto &valueInfo = valueSet.getValueInfo(i);
        if (valueInfo.getFullValueRef(i).isAllPresent(liveness.tracked))
          liveValueIds.push_back(i);
      }
      if (!liveValueIds.empty())
        op.setAttr(liveValueIdsAfterNoReturnCallAttrName,
                   mlir::DenseI32ArrayAttr::get(op.getContext(), liveValueIds));
    }
  }

  // Indicate that this block is no longer live, so no values from it get merged
  // at "if" joins etc.
  liveness.markReachable(false);
}

/// This is HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, which all
/// perform local control flow.
void UninitializedValueScan::checkLocalControlFlowOp(Operation &op) {
  if (isa<HLCF::BreakOp, ParamForBreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    breakSet->mergeWith(liveness, valueSet.domInfo);
  } else if (isa<HLCF::ContinueOp, ParamForContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    continueSet->mergeWith(liveness, valueSet.domInfo);
  } else {
    StringAttr label = cast<LIT::TryRaiseOp>(op).getLabelAttr();
    RaiseSetEntry<TrackedAndInteriorLiveness> *matchingSet =
        raiseEntryInfo->getMatchingRaiseSet(label);
    //  lower-semantic-cf should guarantee there is a matching set.
    assert(matchingSet && "No matching 'try'?");
    // Only merges the set with the matching label.
    matchingSet->raiseSet->mergeWith(liveness, valueSet.domInfo);
  }

  // Indicate that all values are live after the terminator so an 'if' will get
  // properly intersected with the other side of the branch.
  liveness.markReachable(false);
}

/// This is HLCF::IfOp, ParamIfOp, or a simple (no extra arms) HLCF::ElifOp.
void UninitializedValueScan::checkIfLikeOp(Operation &op) {
  // 'if' operations treat the condition as a use but have live outs that are
  // the intersection of the live values produced by the then/else branches.
  assert((isa<HLCF::IfOp, ParamIfOp, HLCF::ElifOp>(op)));
  assert(op.getNumRegions() == 2 && op.getRegion(0).hasOneBlock() &&
         op.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");

  TrackedAndInteriorLiveness livenessCopy = liveness;
  scanBlock(op.getRegion(0).front());
  livenessCopy.swap(liveness);
  scanBlock(op.getRegion(1).front());
  liveness.mergeWith(livenessCopy, valueSet.domInfo);
}

// This is used for the HLCF::ElifOp.
void UninitializedValueScan::checkElIfOp(HLCF::ElifOp op) {
  // Region layout: thenRegion, elseRegion, then additional (cond, then) pairs
  // in elifRegions. Live-out is the intersection of all then arms and else.
  auto thenLiveOutValues =
      TrackedAndInteriorLiveness::getEmptyWithMatchingSize(liveness);
  TrackedAndInteriorLiveness scratchSet(0, 0); // always overwritten below

  // First then uses the live-in set of the elif (cond is an SSA operand).
  scratchSet = liveness;
  scanBlock(op.getThenRegion().front());
  thenLiveOutValues.mergeWith(liveness, valueSet.domInfo);
  std::swap(liveness, scratchSet);

  MutableArrayRef<Region> ifRegions = op.getElifRegions();
  assert((ifRegions.size() % 2) == 0 && "Must have pairs of regions");
  for (size_t nextElIfRegion = 0, e = ifRegions.size(); nextElIfRegion != e;
       nextElIfRegion += 2) {
    // Check the next condition accumulating into liveness.
    scanBlock(ifRegions[nextElIfRegion].front());
    // Save the live set after the condition but before the 'then' block.
    scratchSet = liveness;

    // Scan the "then" block for this condition.
    scanBlock(ifRegions[nextElIfRegion + 1].front());
    thenLiveOutValues.mergeWith(liveness, valueSet.domInfo);

    // Restore the live-in set to the set of things before the 'then' block.
    std::swap(liveness, scratchSet);
  }

  // After each of the cases has been evaluated, check the 'else' block.
  scanBlock(op.getElseRegion().front());

  // The live out set of the whole 'elif' is the intersection of the output set
  // of the else as well as all the 'then' blocks.
  liveness.mergeWith(thenLiveOutValues, valueSet.domInfo);
}

void UninitializedValueScan::checkLoopOp(Operation &loopOp) {
  UninitializedValueScan bodySets(valueSet, interiorOriginTracker);
  // Loops are transparent to raise.
  bodySets.raiseEntryInfo = raiseEntryInfo;

  // The default continueSet is the live-in set of values.  This can lose
  // values if some 'continue' path through the body of the loop consumes a
  // value.
  TrackedAndInteriorLiveness continueSet = liveness;
  bodySets.continueSet = &continueSet;

  // The 'breakSet' of the loop body will be the live outs of the loop.
  auto breakSet =
      TrackedAndInteriorLiveness::getEmptyWithMatchingSize(liveness);
  bodySets.breakSet = &breakSet;

  // Iteratively scan the loop body until the live-in set converges.  This is
  // a trivial lattice with each bit converging to "not live in", so we know
  // this will terminate.
  std::pair<size_t, size_t> numLiveIn;
  do {
    numLiveIn = continueSet.countNumLive();
    // Scan the body: any breaks will intersect their live-out set with
    // 'breakSet', and any continues will intersect their live-out set with
    // 'continueSet'.
    bodySets.liveness = continueSet;
    bodySets.scanBlock(loopOp.getRegion(0).front());

    // If any bits got cleared from the continueSet then we need to iterate.
  } while (continueSet.countNumLive() != numLiveIn);
  // Any code after the loop continues on with the breaks valid.

  // If the loop has an 'else' region, scan it and then intersect with the loop
  // region.  ParamForLoopOp's will have an 'unreachable' in the else region
  // because LowerSemanticCF already processed them.
  if (loopOp.getNumRegions() == 2 && !isa<ParamForOp>(loopOp)) {
    scanBlock(loopOp.getRegion(1).front());
    liveness.mergeWith(breakSet, valueSet.domInfo);
  } else {
    liveness = std::move(breakSet);
  }
}

void UninitializedValueScan::checkTryOp(LIT::TryOp tryOp) {
  UninitializedValueScan bodySets(valueSet, interiorOriginTracker);
  // Our current live-in set is live-in to the try body.
  bodySets.liveness = liveness;

  // Try is transparent to break/continue.
  bodySets.continueSet = continueSet;
  bodySets.breakSet = breakSet;

  // We capture all the common values live-out of raise's as being the live-in
  // to the except block.
  auto exceptSet =
      TrackedAndInteriorLiveness::getEmptyWithMatchingSize(liveness);
  // Attach a new entry to the try scope, such that the inner try op only merge
  // the exceptSet with the matching label.
  RaiseSetEntry<TrackedAndInteriorLiveness> exceptInfo = {
      tryOp.getLabelAttr(),
      &exceptSet,
      raiseEntryInfo,
  };
  bodySets.raiseEntryInfo = &exceptInfo;
  bodySets.scanBlock(tryOp.getTryRegion().front());

  // The live-ins to the except block are the exceptSet.
  liveness = std::move(exceptSet);
  scanBlock(tryOp.getExceptRegion().front());

  // The live-out set of the bodySet is the live-in to the else block, but
  // exceptions raised in it go out of the try.
  bodySets.raiseEntryInfo = raiseEntryInfo;
  bodySets.scanBlock(tryOp.getElseRegion().front());

  // The fall through live values are the intersection from the except and
  // else blocks.
  liveness.mergeWith(bodySets.liveness, valueSet.domInfo);
}

//===----------------------------------------------------------------------===//
// DestructorInserter
//===----------------------------------------------------------------------===//

/// Strip RefLoadOp/RebindOp sugar to find the underlying var decl, if any.
/// RefLoadOp can only be used on register passable values, and a `comptime`
/// type alias lowers to a RebindOp wrapping the aliased value; both need to
/// be looked through so that lifetime start/end markers agree on which
/// value they refer to.
static Value stripToVarDeclLookThrough(Value value) {
  if (auto load = value.getDefiningOp<RefLoadOp>())
    value = load.getOperand();
  return RefImmutOp::stripRebinds(value);
}

/// Emit a origin end marker for a value that is being consumed.
static void emitLifetimeEnd(Value value, ImplicitLocOpBuilder &builder) {
  value = stripToVarDeclLookThrough(value);

  if (value.getDefiningOp<VarDeclOp>())
    VarLifetimeEndOp::create(builder, value);
}

static void emitLifetimeEndAfter(Value value, Operation *after) {
  ImplicitLocOpBuilder builder(after->getLoc(), after);
  builder.setInsertionPointAfter(after);
  emitLifetimeEnd(value, builder);
}

namespace {
/// This class holds transient state for the DestructionInsertion pass,
/// accumulating values that need to be destroyed and then emitting and
/// scheduling the destructor calls themselves (potentially mutating the
/// operation with the uses (eg if it is a copyinit).
class DestructorInserter {
public:
  DestructorInserter(ImplicitLocOpBuilder builder, ValueSet &valueSet,
                     std::vector<InFlightDiagnostic> &diagsToEmit)
      : builder(builder), valueSet(valueSet), diagsToEmit(diagsToEmit) {}

  /// This method indicates that the specified value needs to be destroyed after
  /// this operation.  If 'fieldsToDestroy' is non-empty then it specifies which
  /// subfields should be destroyed with zeros, otherwise the whole value needs
  /// to be destroyed.
  void add(Value value, ValueRef valueRef, BitVector fieldsToDestroy = {}) {
    // Look through lit.ref.immut ops to find the underlying mutable thing if
    // we can.  This also helps copy elision which checks for pointer identity.
    value = RefImmutOp::strip(value);
    valuesToDestroy.push_back({value, valueRef, std::move(fieldsToDestroy)});
  }

  enum class DtorEmissionResult {
    /// The destructors were emitted as normal.
    KeepOp,
    /// The operation has been subsumed by a destructor and should be removed.
    RemoveOpWithUse,
  };

  /// This emits any destructors needed at the location specified by the
  /// builder.  If opWithUse is specified, then the inserter is allowed to
  /// perform various optimizations, e.g. if the opWithUse is a copyinit.
  ///
  /// This returns an enum indicating what to do with opWithUse, e.g. if it is
  /// to be deleted by the caller.
  DtorEmissionResult emitDestructors(Operation *opWithUse);

  /// The same as emitDestructors, but there is no opWithUse so no copyinit
  /// elision can happen.
  void emitDestructors() {
    auto result = emitDestructors(/*opWithUse*/ nullptr);
    assert(result == DtorEmissionResult::KeepOp &&
           "should never delete an op if one isn't provided");
    (void)result;
  }

  LLVM_DUMP_METHOD void dump() const;

  /// This is the builder used to insert any destructor calls.
  ImplicitLocOpBuilder builder;

  /// Emit a new diagnostic error if this value has not yet been diagnosed.
  template <typename... Args>
  InFlightDiagnostic *emitErrorIfNotDiagnosed(ValueInfo &valueInfo,
                                              Args &&...args) {
    auto diagOr =
        valueInfo.emitErrorIfNotDiagnosed(std::forward<Args>(args)...);
    if (!diagOr)
      return nullptr;
    return &diagsToEmit.emplace_back(std::move(*diagOr));
  }

  template <typename... Args>
  InFlightDiagnostic &emitWarning(Args &&...args) {
    return diagsToEmit.emplace_back(
        mlir::emitWarning(std::forward<Args>(args)...));
  }

private:
  ValueSet &valueSet;

  /// This is a set of warnings to emit from this pass.  We buffer them and then
  /// emit them at the end of the pass, because dtor insertion is "bottom up"
  /// and we want to emit warnings in a "top down" manner.
  std::vector<InFlightDiagnostic> &diagsToEmit;

  /// During the core op-processing loop, this is the set of values that need to
  /// be destroyed.
  struct ValueToDestroy {
    /// This the SSA value that needs to be destroyed.
    Value value;
    /// The field range covered by value.
    ValueRef valueRef;
    /// If not zero length, this indicates that some subfields are already dead
    /// and the rest need to be destroyed.
    BitVector fieldsToDestroy;
  };
  SmallVector<ValueToDestroy> valuesToDestroy;

  void destroyValueIfNeeded(Value v, ValueRef valueRef,
                            const BitVector &consumedValues,
                            ImplicitLocOpBuilder &builder);
  void emitDestructorCall(Value value, ValueRef valueRef,
                          ImplicitLocOpBuilder &builder);
  DtorEmissionResult optimizeCopyDestroys(Operation *opWithUse);

  enum class CopyInitSuccess {
    Failed,          // Failed to elide.
    Eliminated,      // Eliminated the copyinit entirely.
    ConvertedToMove, // Instruction is still now a moveinit
  };
  CopyInitSuccess elideCopyInitMem(LIT::CallOp copyInitCall, Value copyInitSrc);
  void elideCopyInitReg(LIT::CallOp copyInitCall, Value copyInitSrc);
};
} // end anonymous namespace

void DestructorInserter::dump() const {
  auto &os = llvm::errs();
  os << "Destructor inserter with " << valuesToDestroy.size() << " values\n";
  for (auto &elt : valuesToDestroy)
    os << "  id #" << elt.valueRef.valueId << ": " << elt.value << "\n";
}

/// This emits any destructors needed at the location specified by the
/// builder.  If opWithUse is specified, then the inserter is allowed to
/// perform various optimizations, e.g. if the opWithUse is a copyinit.
///
/// This returns an enum indicating what to do with opWithUse, e.g. if it is
/// to be deleted by the caller.
DestructorInserter::DtorEmissionResult
DestructorInserter::emitDestructors(Operation *opWithUse) {
  // Exit if there is nothing to do.
  if (valuesToDestroy.empty())
    return DtorEmissionResult::KeepOp;

  // If this is a copy ctor call, we can do elision, which may subsume
  // one of our dtors that we need to emit.
  DtorEmissionResult removedOp = optimizeCopyDestroys(opWithUse);

  // There can be dependencies between dtor calls (e.g. an array of references
  // needs to be destroyed before the elements it references). However, values
  // are discovered and numbered in dominance order, so we know we can have a
  // stable order between things destroyed at the same time.
  if (valuesToDestroy.size() > 1) {
    // This sorts in reverse order because emitters insert code at the top.
    std::stable_sort(valuesToDestroy.begin(), valuesToDestroy.end(),
                     [](const ValueToDestroy &a, const ValueToDestroy &b) {
                       return a.valueRef.valueId > b.valueRef.valueId;
                     });
  }

  // Emit each value destruction in turn.
  for (auto &v : valuesToDestroy)
    destroyValueIfNeeded(v.value, v.valueRef, v.fieldsToDestroy, builder);

  // Now that we're done, recycle our space for the next iteration.
  valuesToDestroy.clear();
  return removedOp;
}

/// We need to destroy the specified value, which could destroyed as a single
/// destructor call, or could need fieldwise destruction.  Emit the necessary
/// element accesses and calls.
void DestructorInserter::destroyValueIfNeeded(Value value, ValueRef valueRef,
                                              const BitVector &consumedValues,
                                              ImplicitLocOpBuilder &builder) {

  // If we've recursed down to a field that is already fully destroyed, then
  // we're done without further investigation.
  if (!consumedValues.empty() && valueRef.isAllPresent(consumedValues))
    return;

  // If the entire value needs to be destroyed, then emit a destructor for the
  // whole value.  This is the base case for our recursion.
  if (consumedValues.empty() || !consumedValues.test(valueRef.endBit - 1)) {
    // Diagnose an error if a field of the value we must destroy is already
    // destroyed.  We cannot run the destructor on the whole object if one of
    // the fields is missing.
    if (!consumedValues.empty() && !valueRef.isAllMissing(consumedValues)) {
      auto *diagOr = emitErrorIfNotDiagnosed(
          valueSet.getValueInfo(valueRef.valueId), builder.getLoc(), "field ");
      if (!diagOr)
        return;
      auto &diag = *diagOr;

      auto aliveValues = consumedValues;
      aliveValues.flip();
      // If some fields are present and others are missing, complain about the
      // first whole field that is missing.
      addBadValueNameToDiag(valueRef, aliveValues, valueSet, diag);
      diag << " destroyed out of the middle of a value, preventing the overall "
              "value from being destroyed";
      return;
    }

    // Ok, the value needs to be dead here.  If we're tracking it and this is
    // a whole object destroy, emit a debug kill (unless we're extending the
    // debug lifetime of trivially-destructible types at -O0).
    if (valueRef.valueId) {
      const ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
      if (info.debugVariable &&
          (consumedValues.empty() ||
           valueRef.getNumBits() == consumedValues.size()) &&
          !valueSet.shouldSuppressDebugKill(info)) {
        DebugInfo::KillOp::create(builder, info.debugVariable);
      }
    }

    // Emit the destructor.
    emitDestructorCall(value, valueRef, builder);
    return;
  }

  // Otherwise, we must have an indirect value where some fields are present and
  // some are missing.  Recursively walk the type and destroy just the fields
  // that are missing.
  auto valueType = sugarCast<LIT::StructType>(valueRef.getValueType(value));
  LIT::StructDeclOp structDecl =
      valueSet.getTypeDeclInfo().getStructInfoForType(valueType).decl;

  // Initialize an evaluator so that we can resolve the field types.
  ParameterEvaluator evaluator;
  for (auto [decl, value] :
       llvm::zip(structDecl.getParams(), valueType.getParamValues()))
    evaluator.setDeclBinding(decl, value);

  assert(valueRef.isIndirect && "register values aren't field sensitive");

  unsigned nextBit = 0;
  for (StructFieldOp field : structDecl.getFieldDecls()) {
    auto fieldVal = RefStructGEROp::create(builder, value, field);
    unsigned numBits = valueSet.getTypeDeclInfo().getNumFieldsInType(
        evaluator.getReboundType(field.getType()));
    destroyValueIfNeeded(fieldVal, valueRef.getSubfield(nextBit, numBits),
                         consumedValues, builder);

    // If there was no destructor generated (because the element has no
    // destructor) then remove the unused pointer access.
    if (fieldVal->use_empty())
      fieldVal->erase();
    nextBit += numBits;
  }
  // The whole object bit should exist after all the fields.
  assert(valueRef.startBit + nextBit + 1 == valueRef.endBit &&
         "Lost track of bits");
}

/// Given a value of reference type, this checks to see if it is immutable, and
/// casts it back to a mutable reference.  This isn't a generally safe operation
/// from a type system perspective, so should only be used for things like
/// destructor insertion that happen after borrow checking.
static Value getMutableRefForPossiblyImmutValue(Value value,
                                                ImplicitLocOpBuilder &builder) {
  value = RefImmutOp::strip(value);

  // Check to see if the reference is already mutable.
  auto destType = cast<RefType>(value.getType()).getWithMutability(true);
  if (value.getType() == destType)
    return value;

  return RebindOp::create(builder, destType, value);
}

/// Emit one destructor call for one entire value or field.
///
/// The 'opWithUse' value, if present, is the operation using the overall value
/// being destroyed.  This allows us to perform copy ctor+temp elision.
void DestructorInserter::emitDestructorCall(Value value, ValueRef valueRef,
                                            ImplicitLocOpBuilder &builder) {
  Type destroyedType =
      ValueRef::getDereferencedType(value.getType(), valueRef.isIndirect);

  SpecialMemberInfo dtorInfo = valueSet.getTypeDeclInfo().getDestructorForType(
      destroyedType, valueSet.getFunc(), builder.getLoc());

  // If there is no destructor, then this is either a trivial type or a
  // linear type.  If linear, emit the error message.
  if (dtorInfo.isUnavailable()) {
    ValueInfo &valueInfo = valueSet.getValueInfo(valueRef.valueId);
    auto diagOr = valueInfo.emitErrorIfNotDiagnosed(builder.getLoc());
    if (!diagOr)
      return;
    auto &diag = *diagOr;

    // If we have a tracked value with a name, quote it.  Otherwise, for
    // indirect mutable assignments like `ptr[] = linear^` the LHS isn't a
    // simple identifier and has no user-visible name -- describe the
    // abandoned target generically and let the source location point at the
    // offending expression.
    if (valueRef.valueId)
      diag << "'" << valueInfo.getName().str() << "'";
    else
      diag << "value";

    diag << " abandoned without being explicitly destroyed: "
         << dtorInfo.getMessageIfUnavailable().strref();

    // If the insertion point is on a mark_consumed op inserted as part of an
    // exception throw, then add a note explaining this is due to a thrown
    // error.
    Block &block = *builder.getInsertionBlock();
    if (builder.getInsertionPoint() != block.end() &&
        isa<LIT::OwnershipMarkConsumedOp>(*builder.getInsertionPoint()) &&
        isa<ErrorReturnOp, TryRaiseOp>(block.getTerminator())) {
      diag.attachNote(builder.getLoc())
          << "value was not consumed when an error is thrown";
    }

    // If the value is a parameter of trait type, then that parameter needs to
    // add a Deinitable trait conformance.
    if (auto generic = sugarDynCast<ParamType>(destroyedType)) {
      if (auto trait = sugarDynCast<TraitType>(generic.getParam().getType())) {
        // TODO: We should really be able to use ASTPrinter.cpp here, need to
        // sink it to LIT dialect support though.
        diag.attachNote(builder.getLoc())
            << "consider adding trait conformance to Deinitable";
      }
    }
    return;
  }

  TypedAttr dtor = dtorInfo.getMember();

  // Emit the marker to denote the end of lifetime for types with trivial dtors.
  if (!dtor)
    return emitLifetimeEnd(value, builder);

  FuncType signature = cast<FuncTypeGeneratorType>(dtor.getType()).getBody();
  assert(signature.getNumResults() == 1 &&
         "dtor should have one result (none type)");
  assert(signature.getNumArguments() == 1 && "dtor should have one operand");

  // We may have a RegisterPassable value direct (e.g. because it is not in a
  // var).  If so, it needs to be stored into a temporary to invoke the
  // destructor, because it takes it by-ref.
  if (!isa<RefType>(value.getType())) {
    size_t originNum = valueSet.nextCounterValue();
    StringAttr originAttr =
        builder.getStringAttr("__dtor_tmp__`" + Twine(originNum));
    auto tmpVar = VarDeclOp::create(
        builder, value.getType(),
        builder.getStringAttr("__dtor_tmp__" + Twine(originNum)), originAttr,
        VarDeclKind::Implicit);
    VarLifetimeStartOp::create(builder, tmpVar);
    RefStoreOp::create(builder, value, tmpVar);
    value = tmpVar;
  }

  // The dtor must take a reference:  Bind the implicit origin of __del__'s self
  // to the origin of the reference we have.
  SmallVector<TypedAttr> implicitOrigins;
  auto delSelfTy = dyn_cast<RefType>(signature.getArgument(0));
  if (!delSelfTy) {
    emitErrorIfNotDiagnosed(
        valueSet.getValueInfo(valueRef.valueId), builder.getLoc(),
        "invalid __del__ that doesn't take register by-ref");
    return;
  }

  value = getMutableRefForPossiblyImmutValue(value, builder);
  auto argRef = cast<RefType>(value.getType());

  implicitOrigins.push_back(argRef.getOrigin());

  // Verify that the address space of the reference matches.  The __del__
  // method will have address space zero.  Attempts to delete other things
  // should not explode the compiler.
  if (delSelfTy.getAddressSpace() != argRef.getAddressSpace()) {
    emitErrorIfNotDiagnosed(
        valueSet.getValueInfo(valueRef.valueId), builder.getLoc(),
        "cannot destroy value in non-default address space");
    return;
  }

  // Adjust element type sugar if needed.
  // FIXME(MOCO-3006): Re-enable / adjust this assertion: we need to compare the
  // metatypes of the types, not the types themselves.  Metatype equivalence is
  // how we can look through type parameters to see their trait bounds match
  // correctly.  How do we do this in KGEN?
  // assert(isEqualCanon(delSelfTy.getElementType(), argRef.getElementType()));
  if (delSelfTy.getElementType() != argRef.getElementType()) {
    value = RebindOp::create(
        builder, argRef.getWithElement(delSelfTy.getElementType()), value);
    argRef = cast<RefType>(value.getType());
  }

  // Emit the call to the destructor.
  LIT::CallOp::create(builder, signature.getResults()[0], dtor, implicitOrigins,
                      value);
  emitLifetimeEnd(value, builder);
}

//===----------------------------------------------------------------------===//
// DestructorInserter Copy Elision
//===----------------------------------------------------------------------===//

/// Look to see if the specified operation is a copyinit: if so, check to see
/// if any of the values we're looking to destroy are the input.  If so, try to
/// eliminate the copy in favor of more uses of the now-dead input.
///
///   %tmp = lit.var.decl "anonymous"
///   kgen.call __init__(copy=)(%src, %tmp)
///   kgen.call __del__(%src)   <<= Thinking about inserting this.
///   kgen.call user(%tmp)      <<= Consuming call.
///
/// If this happens, we want to generate:
///    REMOVED: %tmp = lit.var.decl "anonymous"
///    REMOVED: kgen.call __init__(copy=)(%src, %tmp)
///    NOTADDED: kgen.call __del__(%src)
///    kgen.call user(%src)      <<= Use %src instead.
///
/// Similar, for a register form, we want to transform:
///    %tmp = kgen.call __init__(copy=)(%src)
///    kgen.call __del__(%src)   <<= Thinking about inserting this.
///    ...
///    lit.ref.store %tmp, %copy
///    ...
///    kgen.call user(%copy)      <<= Consuming call.
///
/// Into:
///    %tmp = lit.ref.load %src
///    ...
///    lit.ref.store %tmp, %copy
///    ...
///    kgen.call user(%copy)      <<= using call.
DestructorInserter::DtorEmissionResult
DestructorInserter::optimizeCopyDestroys(Operation *opWithUse) {
  auto copyInitCall = dyn_cast_if_present<LIT::CallOp>(opWithUse);
  if (!copyInitCall)
    return DtorEmissionResult::KeepOp;

  // See if we can resolve the callee.
  FnOp callee = valueSet.getTypeDeclInfo().getFuncForSymbol(
      copyInitCall.getDirectCallee());
  if (!callee ||
      callee.getSpecialFunctionKind() != SpecialFunctionKind::kCopyCtor)
    return DtorEmissionResult::KeepOp;

  // Register passable and memory types both pass the source in memory.
  Value copySrcMem = RefImmutOp::strip(copyInitCall.getOperand(0));

  // These optimizations are only valid if the type is implicitly destructible.
  // Linear types shouldn't "optimize things into working", they should
  // predictably generate an error message when an implicit destructor (such as
  // what we're processing in this code) is needed.
  auto dtorInfo = valueSet.getTypeDeclInfo().getDestructorForType(
      cast<RefType>(copySrcMem.getType()).getElementType(), valueSet.getFunc(),
      opWithUse->getLoc());
  if (dtorInfo.isUnavailable())
    return DtorEmissionResult::KeepOp;

  // Check to see if the copy is immediately destroyed.  If so, we can elide
  // both the copy and the destroy.
  // NOTE: There is a corner case here to be aware of: the copyinit could be
  // the last use of dest (if the result of the copy is dead) the last use of
  // src (what you'd normally think of) as well as the last use of many other
  // values when the input is a reference with an origin set containing
  // multiple things.  We prefer to delete the copy entirely if we can.

  // Handle the register form: `__init__(*, copy: src) -> T`.
  if (copyInitCall.getNumOperands() == 1) {
    assert(copyInitCall.getCalleeType().getArgConvention(0) ==
               ArgConvention::ImmMem &&
           "non-trivial register types passed in memory");
    ValueToDestroy *deadSrc = nullptr;
    Value copyDst = copyInitCall.getResult(0);
    for (auto [i, elt] : llvm::enumerate(valuesToDestroy)) {
      if (!elt.fieldsToDestroy.empty())
        continue; // Can only optimize full object destructions.

      // Check to see if the destination is unused.  If so, we can just drop the
      // copy ctor entirely.
      if (elt.value == copyDst && !elt.valueRef.isIndirect) {
        Value immSrc = copyInitCall.getOperand(0); // src as immutable reference
        copyInitCall->dropAllReferences();
        valueSet.eraseValueInfo(copyDst);

        // If the input was a lit.ref.immut that is now dead, clean it up.
        if (immSrc.use_empty()) {
          if (auto immut = immSrc.getDefiningOp<RefImmutOp>())
            immut->erase();
        }

        // We're done with this destructor, so remove it from the list.
        valuesToDestroy.erase(valuesToDestroy.begin() + i);
        // Caller will remove the copyinit call.
        return DtorEmissionResult::RemoveOpWithUse;
      }

      // Check to see the copy is the last use of the src value.  If so we can
      // always use the source, optimizing a copy to a move.
      if (elt.value == copySrcMem && elt.valueRef.isIndirect)
        deadSrc = &elt;
    }

    // If the source is found to be dead, eliminate it.
    if (deadSrc) {
      elideCopyInitReg(copyInitCall, copySrcMem);

      // We're done with this destructor, so remove it from the list.
      valuesToDestroy.erase(deadSrc);
      // Caller will remove the copyinit call.
      return DtorEmissionResult::RemoveOpWithUse;
    }

    return DtorEmissionResult::KeepOp;
  }

  // Otherwise we have the memory form of `__init__(copy: T, dest: T)`.
  Value copyDstMem = copyInitCall.getOperand(1);

  // Check to see if the destination is unused.  If so, we can just drop the
  // copy ctor entirely.  We need to do this before checking to see if the
  // source is dead.
  ValueToDestroy *deadSrc = nullptr;
  for (auto [i, elt] : llvm::enumerate(valuesToDestroy)) {
    if (!elt.fieldsToDestroy.empty())
      continue; // Can only optimize full object destructions.

    if (elt.value == copyDstMem && elt.valueRef.isIndirect) {
      copyInitCall->dropAllReferences();
      emitLifetimeEndAfter(copyDstMem, copyInitCall);

      // We're done with this destructor, so remove it from the list.
      valuesToDestroy.erase(valuesToDestroy.begin() + i);
      // Caller will remove the copyinit call.
      return DtorEmissionResult::RemoveOpWithUse;
    }

    // Check to see the copy is the last use of the src value, if so, try to
    // use the src directly instead of copying it.
    if (elt.value == copySrcMem && elt.valueRef.isIndirect)
      deadSrc = &elt;
  }

  // If the entire copy isn't dead, but the source is dead, then we can remove
  // it.
  if (deadSrc) {
    DtorEmissionResult result = DtorEmissionResult::KeepOp;
    switch (elideCopyInitMem(copyInitCall, copySrcMem)) {
    case CopyInitSuccess::Failed:
      return DtorEmissionResult::KeepOp;
      // Couldn't elide anything.
    case CopyInitSuccess::Eliminated:
      result = DtorEmissionResult::RemoveOpWithUse;
      break;
    case CopyInitSuccess::ConvertedToMove:
      break; // Remove the dtor, but keep the move.
    }

    // We're done with this destructor, so remove it from the list.
    valuesToDestroy.erase(deadSrc); // valuesToDestroy.begin() + i);
    return result;
  }

  return DtorEmissionResult::KeepOp;
}

/// Return true if the specified 'p1' pointer could point at object or a
/// subcomponent of 'p2'.  This should return true conservatively.
// TODO: In the presence of returned references / origins, we will
// need to be more careful here.
static bool mightPointTo(Value p1, Value p2) {
  assert((isa<PointerType, RefType>(p2.getType())));
  // If the value is an integer or other random thing, then it can't point to
  // anything.
  if (!isa<PointerType, RefType>(p1.getType()))
    return false;

  Value underlyingP1 = OriginTrackable::findUnderlyingValueFromField(p1);
  Value underlyingP2 = OriginTrackable::findUnderlyingValueFromField(p2);
  return !underlyingP1 || !underlyingP2 || underlyingP1 == underlyingP2;
}

// Check to see if we can eliminate a temporary being passed as an owned
// argument to a call.
//
// We currently only do this transformation in extremely limited cases: we
// need to defend against weird situations where "src" doesn't dominate
// "tmp" and where "src" gets mutated before the use of "tmp", e.g.:
//
//    %tmp = lit.var.decl "anonymous"
//    kgen.call __init__(copy=)(%src, %tmp)  <<== Last use of %src
// ** kgen.call __del__(%src)   <<== Thinking about inserting this.
//    kgen.call __init__(%src)  <<== Could reinitialize %src before use of %tmp!
//    use(%tmp) use(%src)
//
// Doing this right requires non-trivial liveness analysis which should
// itself be part of a standalone SSA pass post-inlining.  For now we'll
// just catch the most obvious local cases to clean up the IR and provide a
// "guaranteed" optimization.
static bool canEntirelyElideMemoryTemporary(LIT::CallOp copyInitCall,
                                            VarDeclOp tmpDecl) {
  assert(copyInitCall.getOperand(1) == tmpDecl &&
         "the vardecl is known to be directly assigned");
  // Right now we require them to be in the same block, this is overly
  // conservative.
  Block *tmpBlock = tmpDecl->getBlock();
  if (copyInitCall->getBlock() != tmpBlock)
    return false;

  // Find all users of "tmp".
  SmallPtrSet<Operation *, 3> userOfTmp;
  // Worklist of projections of the tmp VarDecl we need to check.
  SmallVector<Value> valuesToCheck;
  valuesToCheck.push_back(tmpDecl);

  while (!valuesToCheck.empty()) {
    Value checkVal = valuesToCheck.pop_back_val();

    for (OpOperand &operand : checkVal.getUses()) {
      Operation *user = operand.getOwner();

      // Ignore lifetime markers.
      if (isa<VarLifetimeStartOp, VarLifetimeEndOp>(user))
        continue;

      if (user->getBlock() != tmpBlock)
        return false; // We don't handle control flow.

      // If we see a lit.ref.immut or rebind of the origin, check all its uses
      // as well.
      if (isa<RefImmutOp, RebindOp, RefUpcastOp>(user)) {
        valuesToCheck.push_back(user->getResult(0));
        continue;
      }

      // Ignore the copyinit of tmp.
      if (user == copyInitCall)
        continue;

      // It may be a lit.load.consume if the value is a register passable type.
      if (auto load = dyn_cast<LoadConsumeOp>(user)) {
        userOfTmp.insert(load);
        continue;
      }

      // Otherwise, the only sort of user we can support is a call.
      auto callUser = dyn_cast<LIT::CallOp>(user);
      if (!callUser)
        return false; // Unknown user.

      // The argument convention for the callee must be consuming or read, not
      // initializing or anything else.  Allowing 'read' allows us to enable
      // this pattern:

      //    %tmp = lit.var.decl "anonymous"
      //    kgen.call __init__(copy=)(%src, %tmp)  <<== Last use of %src
      // ** kgen.call __del__(%src)   <<== Thinking about inserting this.
      //    use(%tmp)       <= use the temp
      //    consume(%tmp)   <= eventually consume it.
      auto convention = callUser.getCalleeType().getBody().getArgConvention(
          operand.getOperandNumber());
      if (convention != ArgConvention::OwnedMem &&
          convention != ArgConvention::DeinitMem &&
          convention != ArgConvention::ImmMem)
        return false;
      userOfTmp.insert(callUser);
    }
  }

  // There have to be usersOfTmp: we check to see if the copy is dead before
  // considering this optimization, so the copy itself can't also be dead.
  assert(!userOfTmp.empty() && "tmp should at least be destroyed");

  // Okay, we only see users of the 'tmp' decl that we can understand.  Do a
  // lexical scan to make sure there is nothing between the initialization of
  // the tmp and the use of the tmp that might re-use the source.
  Value srcPointer = copyInitCall.getOperand(0);
  for (auto it = ++Block::iterator(copyInitCall), e = tmpBlock->end();; ++it) {
    // If we ran off the end of the block but we didn't see the users, then the
    // copyinit doesn't dominate this use, something weird is going on, bail
    // out.
    if (it == e)
      return false;

    // We don't recurse into regions, so be conservative.
    if (it->getNumRegions())
      return false;

    // Scan all the operands to see if any of them are related to %src.
    if (llvm::any_of(it->getOperands(),
                     [&](Value v) { return v && mightPointTo(v, srcPointer); }))
      return false;

    // If this operation is a known user of tmp, then we might be done scanning.
    if (userOfTmp.erase(&*it)) {
      if (userOfTmp.empty())
        return true;
    }
    // Otherwise, keep looking through the block until we see all the users.
  }
  return true;
}

/// This function handles the case when we see a destructor destroying the src
/// value for a copyinit call.  In these cases, we can just use the source value
/// directly and drop the copy.
void DestructorInserter::elideCopyInitReg(LIT::CallOp copyInitCall,
                                          Value copySrcMem) {
  Value copyDst = copyInitCall.getResult(0);

  // Insert a consuming load after the copyinit (so our dtor walk doesn't
  // see it) that will replace the copy.
  ImplicitLocOpBuilder builder(copyInitCall.getLoc(),
                               &*std::next(Block::iterator(copyInitCall)));
  // TODO: we could get more aggressive and reuse the memory temp when the
  // result is insta-stored if there is some reason to do so.
  auto newResult = LoadConsumeOp::create(builder, copySrcMem);
  emitLifetimeEndAfter(copySrcMem, newResult);

  copyDst.replaceAllUsesWith(newResult);
  Value immSrc = copyInitCall.getOperand(0); // src as immutable reference
  copyInitCall->dropAllReferences();

  // If the input was a lit.ref.immut that is now dead, clean it up.
  if (immSrc.use_empty()) {
    if (auto immut = immSrc.getDefiningOp<RefImmutOp>())
      immut->erase();
  }

  // The value returned by the copyinit is an owned value, update the
  // ValueSet to know that the LoadConsume is the new value for the
  // ValueID.
  ValueRef ref = valueSet.getDirectValueRef(copyDst, /*isDeref*/ false);
  assert(ref.valueId != 0 && "expected to find the copy value");
  ValueInfo &info = valueSet.getValueInfo(ref.valueId);
  assert(info.value == copyDst);
  info.value = newResult;
}

/// We need to destroy the source for the specified call to a memory-only
/// copy ctor call.  Attempt to elide it completely or strength reduce it
/// to a move ctor.  The 'copyInitSrc' value is the src operand with
/// lit.ref.immut instructions stripped off.
DestructorInserter::CopyInitSuccess
DestructorInserter::elideCopyInitMem(LIT::CallOp copyInitCall,
                                     Value copyInitSrc) {
  ImplicitLocOpBuilder builder(copyInitCall.getLoc(), copyInitCall);

  // We prefer to completely delete the copy if it is into a temporary location
  // that we can forward.
  //
  // Note: we currently delete explicitly declared temporaries, not just
  // implicit ones.  This is a policy decision, and we should look into
  // the impact on debug information, but generally one wouldn't want debug
  // information to block optimizations.
  if (VarDeclOp tmpDecl =
          copyInitCall.getOperand(1).getDefiningOp<VarDeclOp>()) {
    if (canEntirelyElideMemoryTemporary(copyInitCall, tmpDecl)) {
      // Insert a declaration of the origin for the tmp we're eliding, we know
      // that VarDeclOp's always declare a unique origin.
      auto refType = cast<RefType>(tmpDecl.getType());
      auto param = cast<ParamDeclRefAttr>(refType.getOrigin());

      // The old reference type used a novel origin.  We need to declare it,
      // and coerce back to it with a rebind.
      ParamDeclareOp::create(builder, ParamDeclAttr::get(param),
                             AnyOriginAttr::get(param.getType()));
      auto refCasted =
          RebindOp::create(builder, tmpDecl.getType(), copyInitSrc);

      // Erase the origin start marker for the temporary. However, keep the
      // origin end markers if the aliased value is a var decl, as they will
      // get "inherited" by the aliased value.
      Value value = OriginTrackable::findUnderlyingValueFromField(refCasted);
      for (Operation *user : llvm::make_early_inc_range(tmpDecl->getUsers())) {
        if (isa<VarLifetimeStartOp>(user)) {
          user->erase();
        } else if (auto end = dyn_cast<VarLifetimeEndOp>(user)) {
          if (value.getDefiningOp<VarDeclOp>())
            end.setOperand(value);
          else
            user->erase();
        }
      }
      tmpDecl.getResult().replaceAllUsesWith(refCasted);

      // We'll delete the copyInit but don't want to invalidate iterators so do
      // later.  Remove the operand uses so we don't see them in later def-use
      // scans, and to make it more obvious when reading IR dumps that these
      // will be gone.
      copyInitCall->dropAllReferences();
      // The caller will remove the copyinit call.
      return CopyInitSuccess::Eliminated;
    }
  }

  auto srcRefType = cast<RefType>(copyInitSrc.getType());
  Type destroyedType = srcRefType.getElementType();

  // Otherwise, try to promote to a __moveinit__ call if present.
  TypedAttr moveCtor = valueSet.getTypeDeclInfo().getMoveInitForType(
      destroyedType, builder.getLoc());
  if (!moveCtor)
    return CopyInitSuccess::Failed;

#ifndef NDEBUG
  // moveCtor must have __moveinit__(out self, deinit existing: Self) type.
  FuncType moveSig = cast<FuncTypeGeneratorType>(moveCtor.getType()).getBody();
  assert(moveSig.getNumArguments() == 2);
  assert(moveSig.getArgConvention(0) == ArgConvention::OwnedMem ||
         moveSig.getArgConvention(0) == ArgConvention::DeinitMem);
  assert(moveSig.getArgConvention(1) == ArgConvention::ByRefResult);
  auto moveArgs = moveSig.getArguments();
  auto moveValue1Ref = cast<RefType>(moveArgs[0]);
  // srcRefType is immutable here because it was passed to a copy.
  assert(cast<RefType>(moveArgs[1]).getElementType() == destroyedType &&
         moveValue1Ref.getElementType() == destroyedType &&
         moveValue1Ref.isMutableKnown(true));

  auto destType = cast<RefType>(copyInitCall.getOperand(1).getType());
  assert(destType.getElementType() == srcRefType.getElementType());
#endif

  // We know that the input is mutable (otherwise it wouldn't be tracked for
  // destruction), get the reference to a mutable type.
  copyInitSrc = getMutableRefForPossiblyImmutValue(copyInitSrc, builder);
  srcRefType = cast<RefType>(copyInitSrc.getType());

  // Switch the source operand, and update the origin associated with it.
  copyInitCall.setOperand(0, copyInitSrc);
  copyInitCall.setImplicitOrigins(
      {srcRefType.getOrigin(), copyInitCall.getImplicitOrigins()[1]});

  // Transform the copy into a move.
  copyInitCall.setCalleeAttr(moveCtor);
  emitLifetimeEndAfter(copyInitSrc, copyInitCall);
  // We don't want to remove the copyinit, it is now our moveinit.
  return CopyInitSuccess::ConvertedToMove;
}

//===----------------------------------------------------------------------===//
// DestructorInsertion Analysis
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements the third pass over a function body, which
/// inserts destructors after the last use of values.
struct DestructorInsertion {
  DestructorInsertion(ValueSet &valueSet) : valueSet(valueSet) {}
  DestructorInsertion(const DestructorInsertion &existing) = delete;
  DestructorInsertion(DestructorInsertion &&existing) = default;

  static DestructorInsertion copy(const DestructorInsertion &existing) {
    DestructorInsertion result(existing.valueSet);
    result.consumedValues = existing.consumedValues;
    result.raiseEntryInfo = existing.raiseEntryInfo;
    result.breakSet = existing.breakSet;
    result.continueSet = existing.continueSet;
    result.dryRun = existing.dryRun;
    return result;
  }

  void scanFunction();
  void scanBlock(Block &body);

  LLVM_DUMP_METHOD void dump() const;

private:
  void checkTerminatorOp(Operation &op);
  void checkLocalControlFlowOp(Operation &op);
  void checkIfLikeOp(Operation &op, SmallVector<ResultEffect> &resultEffects);
  void checkElIfOp(HLCF::ElifOp op, SmallVector<ResultEffect> &resultEffects);
  void checkLoopOp(Operation &loopOp);
  void checkTryOp(LIT::TryOp tryOp);

  BitVector unifyConsumedSets(const BitVector &set1, const BitVector &set2);
  void destroyValuesAtEntryIfNeeded(const BitVector &currentConsumeSet,
                                    Block &block,
                                    const BitVector &fullSetToDestroy,
                                    Location loc);

  void checkConsume(Value value, Operation &op, bool isDeref,
                    DestructorInserter &dtorInserter);
  void checkUse(Value value, bool isDeref, DestructorInserter &dtorInserter);
  void checkDef(Value value, Operation &op, bool isDeref,
                DestructorInserter &dtorInserter);
  void checkOriginAccesses(TypedAttr origin, DestructorInserter &dtorInserter);
  bool scheduleNeededDtors(ValueRef use, DestructorInserter &dtorInserter,
                           Value value = Value());

  /// Emit a debug value for the value if it is tracked with debug info.
  void emitDebugInit(Value value, ValueRef valueRef,
                     ImplicitLocOpBuilder &builder);

  /// This is metadata about all the values we are tracking.
  ValueSet &valueSet;

  /// This is the set of values known to be used below this point, so they
  /// should not be destroyed if there are uses.  Any use of a value /not/ in
  /// this set will be a last use that does get destroyed.
  BitVector consumedValues;

  /// When true, scanning an operation or block will not insert destructors, and
  /// certain invariants don't hold.  This is used when processing loops,
  /// because we need to iterate to a fixed point of values live in from
  /// continue blocks before inserting destructors.
  bool dryRun = false;

  /// When analyzing the body of a try, this stack indicates what a
  /// 'raise' should intersect with. It is indexed with a raise label.
  RaiseSetEntry<BitVector> *raiseEntryInfo = nullptr;

  /// When analyzing the body of a loop, these bitset indicates what a 'break'
  /// or 'continue' should produce based on its consumed value set for the
  /// surrounding loop.
  BitVector *breakSet = nullptr;
  BitVector *continueSet = nullptr;

  /// This is a set of warnings to emit from this pass.  We buffer them and then
  /// emit them at the end of the pass, because dtor insertion is "bottom up"
  /// and we want to emit warnings in a "top down" manner.
  std::vector<InFlightDiagnostic> diagsToEmit;
};
} // namespace

[[maybe_unused]] void DestructorInsertion::dump() const {
  auto &os = llvm::errs();
  if (valueSet.getValueInfos().size() < 32) {
    valueSet.dump();
    os << "\n";
  }

  os << "DestructorInsertion for ";
  valueSet.printFuncName(os);
  if (dryRun)
    os << " [DRYRUN]";
  os << "\n  ";
  valueSet.printBV(consumedValues, os) << "\n";

  RaiseSetEntry<BitVector> *curr = raiseEntryInfo;
  os << " raise: {";
  while (curr) {
    os << curr->label << " : ";
    valueSet.printBV(*curr->raiseSet, os) << "\n";
    curr = curr->prev;
  }
  os << " }";

  if (breakSet) {
    os << " break: ";
    valueSet.printBV(*breakSet, os) << "\n";
  }
  if (continueSet) {
    os << " continue: ";
    valueSet.printBV(*continueSet, os) << "\n";
  }
  os.flush();
}

void DestructorInsertion::scanFunction() {
  consumedValues.resize(valueSet.getNumTotalBits());
  // Slot 0 indicates this block is reachable.  This will be cleared if an
  // 'unreachable' operation is noticed.
  consumedValues.set(0);

  // Scan the body of the function.
  FnOp func = valueSet.getFunc();
  Block &funcBody = func.getBodyRegion().front();
  scanBlock(funcBody);

  // The sentinel tracks reachability. The function entry may not be reachable
  // if the function has a global "comptime assert False".
  if (consumedValues[0]) {
    // If any argument values are unconsumed then they must be unused.
    // Emit their destructor calls at the start of the function by acting as
    // though there is a use.
    for (auto [argValue, conv] :
         llvm::zip(func.getBodyRegion().front().getArguments(),
                   func.getFuncTypeGenerator().getArgConventions())) {
      // Ignore undef-on-input values.
      if (isResultSlot(conv))
        continue;

      bool isIndirect = hasAddress(conv);
      Location loc = argValue.getLoc();
      if (DebugInfo::DISubprogramAttr scope = func.getSubprogramScope())
        loc = FusedLoc::get(loc.getContext(), {loc}, scope);

      ImplicitLocOpBuilder builder(loc, &funcBody, funcBody.begin());
      DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);
      checkUse(argValue, /*isDeref=*/isIndirect, dtorInserter);
      dtorInserter.emitDestructors();
    }
  }

  // Emit any diagnostics that were queued up in a top-down order.
  while (!diagsToEmit.empty())
    diagsToEmit.pop_back();
}

/// Scan a block top down, checking all the operations that may use a value or
/// change its liveness state.  This diagnoses uses of values that are not yet
/// initialized, and returns the set of values that are live at the end of the
/// block.
void DestructorInsertion::scanBlock(Block &block) {
  // Process each operation bottom-up in the block.
  OperationEffects opEffects(valueSet.getOriginFinder());

  SmallVector<Operation *> opsToRemove;

  for (Operation &op : llvm::reverse(block)) {
    auto overall = opEffects.analyze(op);
    switch (overall) {
    case OverallOpValueEffect::unknownOp:
      // NOTE: Enable logging when debugging.
      // op.dump();
      continue;
    case OverallOpValueEffect::allHandled:
      break; // No special action.
    case OverallOpValueEffect::terminatorOp:
      checkTerminatorOp(op);
      break;
    case OverallOpValueEffect::localControlFlowOp:
      checkLocalControlFlowOp(op);
      break;
    case OverallOpValueEffect::ifLikeOp:
      checkIfLikeOp(op, opEffects.results);
      break;
    case OverallOpValueEffect::elifOp: {
      auto elifOp = cast<HLCF::ElifOp>(op);
      // A single if/else shaped elif has the same region layout as IfOp; reuse
      // the proven if-like destructor logic (important inside loops).
      if (elifOp.getElifRegions().empty())
        checkIfLikeOp(op, opEffects.results);
      else
        checkElIfOp(elifOp, opEffects.results);
      break;
    }
    case OverallOpValueEffect::loopOp:
      checkLoopOp(op);
      break;
    case OverallOpValueEffect::tryOp:
      checkTryOp(cast<LIT::TryOp>(op));
      break;
    }

    // Insert any destructor calls immediately /after/ this operation, since
    // they are for values used by it.
    ImplicitLocOpBuilder builder(op.getLoc(), op.getBlock(),
                                 std::next(Block::iterator(&op)));
    DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);

    assert((opEffects.results.size() == op.getNumResults() ||
            overall == OverallOpValueEffect::ifLikeOp ||
            overall == OverallOpValueEffect::elifOp) &&
           "OperationEffects::analyze returned wrong # effects");

    for (auto [result, effect] :
         llvm::zip(op.getResults(), opEffects.results)) {
      // CheckUninit pass does all the paranoid checking, don't duplicate it.
      switch (effect) {
      case ResultEffect::ignore:
        continue;
      case ResultEffect::regDefine:
        checkDef(result, op, /*isDeref=*/false, dtorInserter);
        break;
      case ResultEffect::memDefineUninitToInit:
        // The live-in behavior is modeled by OriginTrackable to match the
        // live-out behavior.
        // We consume on execution to provide Uninit -> Init behavior.
        checkConsume(result, op, /*isDeref=*/true, dtorInserter);
        break;
      case ResultEffect::memDefineUninitToUninit:
        // Nothing to do here.
        break;
      case ResultEffect::memDefineInitToInit:
        // If it start/end initialized, emit destructor if already replaced.
        checkUse(result, /*isDeref=*/true, dtorInserter);
        break;
      case ResultEffect::memDefineInitToUninit:
        // We consume on execution to provide Init -> Uninit behavior.
        checkDef(result, op, /*isDeref=*/true, dtorInserter);
        break;
      }
    }

    // Handle all the normal operand and result effects.
    for (auto [operand, effect] : opEffects.operands) {
      switch (effect) {
      case OperandEffect::regUse:
        checkUse(operand, /*isDeref=*/false, dtorInserter);
        break;
      case OperandEffect::regConsume:
        checkConsume(operand, op, /*isDeref=*/false, dtorInserter);
        break;
      case OperandEffect::memLoad:
        checkUse(operand, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memStoreOwned:
        checkDef(operand, op, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memMut:
        // It is sufficient to just check that we're using the input operation,
        // and if this is the last use of the operation, we should insert a
        // destructor for the value.  checkDef would mark the value as
        // not-live-in, which we don't want.
        checkUse(operand, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memConsume:
        checkConsume(operand, op, /*isDeref=*/true, dtorInserter);
        break;
      case OperandEffect::memMarkDestroyed:
        // If the uninit value pass decided that this mark_destroyed is unused,
        // then we just ignore it.
        if (op.hasAttr(unusedMarkDestroyName))
          break;

        // The lit.ownership.mark_destroyed op consumes the whole object bit of
        // a value only, but not its fields.  This ensures the sub-fields are
        // destroyed but the full object is not.  It is used in destructors
        // primarily.
        if (ValueRef access =
                valueSet.getDirectValueRef(operand, /*isDeref=*/true))
          consumedValues.set(access.endBit - 1);
        break;
      }
    }

    // Process any other origins accessed indirectly.
    for (auto [origin, value] : opEffects.origins) {
      (void)value;
      checkOriginAccesses(origin, dtorInserter);
    }

    // If the operation used a #lit.any.origin value, then we treat it as an
    // implicit use of all tracked values.  This ensures that the values are not
    // destroyed too early.  Uninit variable scan handles this by adding an
    // attribute with all the value ID's in question.
    if (auto extraUses = op.getAttrOfType<mlir::DenseI32ArrayAttr>(
            extraOriginUsesAttrName)) {
      if (!dryRun)
        op.removeAttr(extraOriginUsesAttrName);

      // Treat this op as using each of the indicated values, putting out a
      // destructor call if this is the last use.
      for (int32_t valueId : extraUses.asArrayRef()) {
        const ValueInfo &info = valueSet.getValueInfo(valueId);
        // NOTE: This can be useful to understand what values are getting
        // lifetime extended and by what.  This is intended more for compiler
        // and library development, not for users.
#if 0
        if (!info.getFullValueRef(valueId).isAllPresent(consumedValues)) {
          auto diag = mlir::emitRemark(op.getLoc());
          if (auto call = dyn_cast<LIT::CallOp>(op))
            diag << "call to '" << call.getDirectCallee() << "'";
          else
            diag << "op";

          diag << " extended with AnyOrigin usage extended lifetime of ";
          if (auto varDecl = info.value.getDefiningOp<VarDeclOp>())
            diag << "'" << varDecl.getName() << "'";
          else
            diag << info.value;
        }
#endif

        checkUse(info.value, /*isDeref*/ info.isIndirect, dtorInserter);
      }
    }

    // Finally emit any enqueued destructors.
    if (dtorInserter.emitDestructors(&op) ==
        DestructorInserter::DtorEmissionResult::RemoveOpWithUse) {
      // If we replaced this operation, remove it after our sweep.
      opsToRemove.push_back(&op);
    }
  }

  // If we had any operations to remove, do so now, simplifying iterator
  // invalidation issues.
  for (Operation *op : opsToRemove)
    op->erase();
}

/// This is returned when the op is a return or unreachable op.
void DestructorInsertion::checkTerminatorOp(Operation &op) {
  consumedValues.reset();
  if (auto unreachable = dyn_cast<UnreachableOp>(op)) {
    // If this marker indicates the entire block is unreachable, then we don't
    // mark the block or anything else as live.
    if (!unreachable.getIsAfterUnreachableCall())
      return;

    // If this is after a no-return call, then we do mark all the live-out
    // values (e.g. like "read" arguments) as consumed so they don't get
    // destroyed on this path.  This is calculated on the forward pass.
    if (auto liveValueIds = op.getAttrOfType<mlir::DenseI32ArrayAttr>(
            liveValueIdsAfterNoReturnCallAttrName)) {
      // Mark all the live values as consumed so they don't get destroyed ahead
      // of this.
      for (int32_t valueId : liveValueIds.asArrayRef())
        valueSet.getFullValueRef(valueId).markBits(consumedValues, true);
    }
    return;
  }

  assert((isa<KGEN::ReturnOp, ErrorReturnOp>(op)) && "unknown terminator");
  consumedValues.set(0); // Slot 0 indicates that this block is reachable.

  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    // If this value must be live on exit from the function (e.g. a mut
    // argument) demand it.
    if (!isUninitializedAtExit(valueInfo, op)) {
      consumedValues.set(valueInfo.startValueBit, valueInfo.endValueBit);
      continue;
    }

    // On a throw from an __init__ where self is partially initialized, demand
    // the full object bit of self so we destroy any live fields, but not
    // self as a whole.
    if (isa<ErrorReturnOp>(op) && valueInfo.isFullObjectLiveOnEntry &&
        op.hasAttr(selfPartiallyInitializedAttrName))
      consumedValues.set(valueInfo.endValueBit - 1);
  }
}

void DestructorInsertion::checkLocalControlFlowOp(Operation &op) {
  if (isa<HLCF::BreakOp, ParamForBreakOp>(op)) {
    assert(breakSet && "Not in a loop?");
    consumedValues = *breakSet;
    return;
  }
  if (isa<HLCF::ContinueOp, ParamForContinueOp>(op)) {
    assert(continueSet && "Not in a loop?");
    consumedValues = *continueSet;
    return;
  }

  // A raise will use the consume set that was seen on entry to the enclosing
  // except block.
  StringAttr raiseLabel = cast<LIT::TryRaiseOp>(op).getLabelAttr();
  auto matchingSet = raiseEntryInfo->getMatchingRaiseSet(raiseLabel);
  //  lower-semantic-cf should guarantee there is a matching set.
  assert(matchingSet && "No matching 'try'?");
  consumedValues = *matchingSet->raiseSet;
}

/// 'if' operations propagate the consume sets into each branch, and use the
/// resulting consume sets to make sure the upward propagated set of consumed
/// values is consistent.
void DestructorInsertion::checkIfLikeOp(
    Operation &ifElseOp, SmallVector<ResultEffect> &resultEffects) {

  // If there are any result effects, process them first before going into the
  // body.  This happens when the 'if' defines an owned register value, as in:
  //   %x = hlcf.if %cond { } else { }
  //   -> here.
  // If the register result isn't used, for example, we want the dtor inserted
  // below the 'if' and not propagated into the arms of the 'if'.
  if (!resultEffects.empty()) {
    // Insert any dtors after the 'if'.
    ImplicitLocOpBuilder builder(ifElseOp.getLoc(), ifElseOp.getBlock(),
                                 std::next(Block::iterator(&ifElseOp)));
    DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);
    for (auto [result, effect] :
         llvm::zip(ifElseOp.getResults(), resultEffects)) {
      switch (effect) {
      case ResultEffect::ignore:
        continue;
      case ResultEffect::regDefine:
        checkDef(result, ifElseOp, /*isDeref=*/false, dtorInserter);
        break;
      default:
        llvm_unreachable("unknown result effect for 'if'");
      }
    }
    // Handled these, don't reprocess.
    resultEffects.clear();
  }

  // Given an 'if' like operation (normal 'if' statement or parameter if)
  // perform dtor analysis for each side and insert destructors at the top of
  // the blocks to form a common upward-projected consume set.
  assert(ifElseOp.getNumRegions() == 2 && ifElseOp.getRegion(0).hasOneBlock() &&
         ifElseOp.getRegion(1).hasOneBlock() &&
         "if-like op should have two single-block regions");
  BitVector thenConsumedValues = consumedValues;
  scanBlock(ifElseOp.getRegion(0).front());
  // Scan 'else' block.
  thenConsumedValues.swap(consumedValues);
  scanBlock(ifElseOp.getRegion(1).front());

  BitVector merged = unifyConsumedSets(consumedValues, thenConsumedValues);
  if (merged.empty()) // Common case, they are identical.
    return;

  // 'consumedValues' is the current set for the 'else' block, so insert those
  // dtors if needed.
  destroyValuesAtEntryIfNeeded(consumedValues, ifElseOp.getRegion(1).front(),
                               merged, ifElseOp.getLoc());

  // Insert destructors in the 'then' block.
  destroyValuesAtEntryIfNeeded(thenConsumedValues,
                               ifElseOp.getRegion(0).front(), merged,
                               ifElseOp.getLoc());

  // The upward consume set is the union of both sides.
  consumedValues = std::move(merged);
}

// This is used for the HLCF::ElifOp.
void DestructorInsertion::checkElIfOp(
    HLCF::ElifOp op, SmallVector<ResultEffect> &resultEffects) {
  // Handle owned register results of the elif the same way as if-like ops.
  if (!resultEffects.empty()) {
    ImplicitLocOpBuilder builder(op.getLoc(), op->getBlock(),
                                 std::next(Block::iterator(op)));
    DestructorInserter dtorInserter(builder, valueSet, diagsToEmit);
    for (auto [result, effect] : llvm::zip(op.getResults(), resultEffects)) {
      switch (effect) {
      case ResultEffect::ignore:
        continue;
      case ResultEffect::regDefine:
        checkDef(result, *op, /*isDeref=*/false, dtorInserter);
        break;
      default:
        llvm_unreachable("unknown result effect for 'elif'");
      }
    }
    resultEffects.clear();
  }

  // Region layout: thenRegion, elseRegion, then additional (cond, then) pairs.
  // Backward pass: else, then each additional pair, then the first thenRegion.
  MutableArrayRef<Region> ifRegions = op.getElifRegions();
  assert((ifRegions.size() % 2) == 0 && "Must have pairs of regions");

  BitVector thenExitConsumedValues = consumedValues;
  Block *elseBlock = &op.getElseRegion().front();
  scanBlock(*elseBlock);

  for (size_t i = ifRegions.size(); i != 0; i -= 2) {
    Block &condBlock = ifRegions[i - 2].front();
    Block &thenBlock = ifRegions[i - 1].front();

    BitVector elseConsumeSet = std::move(consumedValues);
    consumedValues = thenExitConsumedValues;
    scanBlock(thenBlock);

    BitVector merged = unifyConsumedSets(consumedValues, elseConsumeSet);
    if (!merged.empty()) {
      destroyValuesAtEntryIfNeeded(consumedValues, thenBlock, merged,
                                   op.getLoc());
      destroyValuesAtEntryIfNeeded(elseConsumeSet, *elseBlock, merged,
                                   op.getLoc());
      consumedValues = std::move(merged);
    }

    scanBlock(condBlock);
    // For the next arm up the chain, this condition is the effective else.
    elseBlock = &condBlock;
  }

  // Finally unify the first thenRegion with the remaining "else" path. The
  // first condition is an SSA operand (no cond region to scan).
  {
    Block &thenBlock = op.getThenRegion().front();
    BitVector elseConsumeSet = std::move(consumedValues);
    consumedValues = thenExitConsumedValues;
    scanBlock(thenBlock);

    BitVector merged = unifyConsumedSets(consumedValues, elseConsumeSet);
    if (!merged.empty()) {
      destroyValuesAtEntryIfNeeded(consumedValues, thenBlock, merged,
                                   op.getLoc());
      destroyValuesAtEntryIfNeeded(elseConsumeSet, *elseBlock, merged,
                                   op.getLoc());
      consumedValues = std::move(merged);
    }
  }
}

/// Given two consume sets that correspond to an 'if-like' construct which
/// diverges control flow, compute the union of the two consume sets and return
/// it, or RETURN AN EMPTY BITVECTOR if they are identical.
///
/// Consider:     if cond: use(a) else: use(b)
///
/// In this case, the 'then' block will use "a" and the else block will use "b".
/// This returns the union of both {a,b}.  This union operation is non-trivial
/// in other cases though.
///
BitVector DestructorInsertion::unifyConsumedSets(const BitVector &set1,
                                                 const BitVector &set2) {
  // If they agree already, then there is nothing to do.
  if (set1 == set2)
    return BitVector();

  // We don't want to perform meets with unreachable code (e.g. from `if False:
  // stuff`: if either of the regions is unreachable, then propagate the other
  // one.  This matters because there is no conservative "missing" set for whole
  // object bits.  We use the sentinel's consume bit to know if anything is
  // consumed.
  if (!set1[0]) // If "then" isn't reachable, return "else".
    return set2;
  if (!set2[0]) // If "else" isn't reachable, return "then".
    return set1;

  // Given two consume sets, our upward propagated final set will be the
  // union of both sets.
  BitVector result = set1;
  result |= set2;

  // It is possible that some subfields out of a value that is fully consumed
  // are not demanded.  For example, consider something like:
  //
  //   def test(cond: Bool):
  //     # Tracked as pair.{a,b,overall}
  //     var pair = Pair(a=String(), b=String())
  //
  //     if cond:            # <- consumes pair.{a,overall}, but not pair.b
  //       pair.b = String() # <- overwrites pair.b so it isn't consumed
  //       pair.use()        # <- consumes pair upwards
  //       return            # <- consumes nothing
  //     else:               # <- consumes nothing
  //       return            # <- consumes nothing
  //
  // In this situation we know that "pair overall" is live into to one side
  // and not live into the other side, that we'll need to destroy the whole
  // thing... so the upward-propagated union needs to demand all of
  // pair.{a,b,overall}.  Computing this allows us to rewrite this into:
  //
  //   def test(cond: Bool):
  //     # Tracked as pair.{a,b,overall}
  //     var pair = Pair(a=String(), b=String())
  //
  //     if cond:
  //       pair.b.__del__()  # the body doesn't demand pair.b, so destroy it
  //       pair.b = String()
  //       pair.use()
  //       pair.__del__()    # destroyed after pair.use's last use.
  //     else:
  //       pair.__del__()    # block doesn't demand pair at all.
  //       return
  //
  // If we see this, have the union set demand the whole object so it can be
  // destroyed.
  for (const ValueInfo &valueInfo : valueSet.getValueInfos()) {
    // If the whole-object consume bits agree on both sides, then there is
    // nothing to do.
    if (!valueInfo.isIndirect)
      continue; // Register values have a single bit.

    // If the whole object is already destroyed on both sides, then we don't
    // have to worry about this.  It may be consuming subobjects at a time.
    auto endBit = valueInfo.endValueBit - 1;
    if (set1[endBit] && set2[endBit])
      continue;

    // If any subfields are consumed, then we consume the whole object so the
    // destructor can be run.
    ValueRef ref(/*index*/ 0, valueInfo.startValueBit, valueInfo.endValueBit,
                 valueInfo.isIndirect);
    // If no part of this value is consumed, then ignore it.
    if (ref.isAllMissing(result))
      continue;

    // If this is a merge between 'self' which is not consumed at all on one
    // side, and is consumed a bit on the other side, ignore this and propagate
    // up the simple union.  This happens in error handling scenarios because
    // the error result doesn't demand anything (not even the full object bit)
    // but the other path can demand a partially initialized set of stuff.
    if (valueInfo.isFullObjectLiveOnEntry)
      continue;

    // Otherwise, some part is required, so require the whole thing on both
    // sides so it can be destroyed.
    result.set(valueInfo.startValueBit, valueInfo.endValueBit);
  }

  return result;
}

/// For a loop, we know the consume sets for any break statements, but need
/// to iterate the loop to find the right continue sets to use.
///
/// In terms of form, both standard for and comptime for loops will have their
/// 'else' block removed (merged into their body).
void DestructorInsertion::checkLoopOp(Operation &loopOp) {
  // True if this is a parameter for, false if this is an infinite HLCF::LoopOp.
  [[maybe_unused]] bool isParamFor = isa<ParamForOp>(loopOp);
  assert((!isParamFor ||
          isa<UnreachableOp>(loopOp.getRegion(1).front().front())) &&
         "LowerSemanticCF should have handled this");

  auto loopBodySets = DestructorInsertion::copy(*this);
  // Any 'break's within the loop will produce the consume set for the
  // statement immediately after the loop.  However, comptime for statements
  // may have an 'else' block that break statements skip over. Save the exit
  // set for break statements.
  BitVector breakSet(consumedValues);

  // The original set will be what any 'break' statement sees.
  loopBodySets.breakSet = &breakSet;

  // The continueSet is the set of values consumed upwards from the top of the
  // loop and carried over the loop.
  //
  // We start the set with no values to be consumed, and with sentinel slot #0
  // unset indicating that the continue point isn't reachable.  This will cause
  // the first iteration to propagate values up from the 'break' points to the
  // consume set.
  BitVector continueSet(breakSet.size());
  loopBodySets.continueSet = &continueSet;

  // We need to dry run the body evaluation until we get to a stable
  // continue set.
  loopBodySets.dryRun = true;

  // Iteratively scan the loop body until the continue set converges.
  [[maybe_unused]] unsigned numIters = 0;
  while (true) {
    // Scan the body: any breaks will intersect their live-out set with
    // 'breakSet', and any continues will intersect their live-out set with
    // 'continueSet'.
    loopBodySets.scanBlock(loopOp.getRegion(0).front());

    // If we scanned the body and didn't find any live code, then we know
    // there must not be any break statements in it.  Just consider the
    // continue point reachable for the next iteration.
    if (!loopBodySets.consumedValues[0])
      loopBodySets.consumedValues[0] = true;

    // If the continue set is unchanged, then we converged.
    if (loopBodySets.consumedValues == continueSet)
      break;

    // Otherwise, use the set of values consumed on loop entry as the new
    // continue set.
    auto merged = unifyConsumedSets(continueSet, loopBodySets.consumedValues);
    if (!merged.empty())
      loopBodySets.consumedValues = std::move(merged);
    continueSet = loopBodySets.consumedValues;

    // This should converge trivially as we are setting bits in the continue
    // set, but when we get a consume operator in the future this may be
    // tricky.  Don't fall into an infinite loop on accident.
    ++numIters;
    assert(numIters < 5 && "Loop should converge in a couple iterations");
  }

  // Once we've converged to the right continue set, we can replay one final
  // iteration in execute mode (if the enclosing context is not dryRun mode)
  // to insert destructors.
  if (!dryRun) {
    loopBodySets.dryRun = false;
    loopBodySets.scanBlock(loopOp.getRegion(0).front());
  }

  consumedValues = std::move(loopBodySets.consumedValues);
}

void DestructorInsertion::checkTryOp(LIT::TryOp tryOp) {
  // The except block is processed with a copy of the consumed value set
  // from the bottom of the try.  After processing it, we know what the
  // consumed values are for the exception block.
  auto exceptSets = DestructorInsertion::copy(*this);
  exceptSets.raiseEntryInfo = raiseEntryInfo;

  Region &exceptRegion = tryOp.getExceptRegion();
  exceptSets.scanBlock(exceptRegion.front());

  // The normal flow finishes with the else block, process it to see what
  // the input consumedValues set to the else block is.
  scanBlock(tryOp.getElseRegion().front());

  // Ok, finally we process the try body.  Any 'raise's within the try body
  // use the consumed values set on entry to the except block.
  // Attach current raise set info to the list as we get one scope deeper.
  RaiseSetEntry<BitVector> curInfo = {
      tryOp.getLabelAttr(),
      &exceptSets.consumedValues,
      raiseEntryInfo,
  };
  llvm::SaveAndRestore x(raiseEntryInfo, &curInfo);
  scanBlock(tryOp.getTryRegion().front());
}

// When the specified value is consumed by an operation we know it doesn't need
// to be destroyed above this point.
void DestructorInsertion::checkConsume(Value value, Operation &op, bool isDeref,
                                       DestructorInserter &dtorInserter) {
  ValueRef valueRef = valueSet.getDirectValueRef(value, isDeref);
  // Uninitialized variable tracking already rejects consumes of indirect
  // non-trivial values.
  if (!valueRef)
    return;

  // If this operation is consuming a sub-element of a value that is already
  // marked to be consumed, then it is being used down below.
  //
  // This happens on code like this, for example:
  //   var a = Pair()
  //   _ = a.x^
  //   use(a.x)
  if (!valueRef.isAllMissing(consumedValues)) {
    ValueInfo &valueInfo = valueSet.getValueInfo(valueRef.valueId);
    auto diagOr =
        dtorInserter.emitErrorIfNotDiagnosed(valueInfo, op.getLoc(), "value ");
    if (!diagOr)
      return;
    auto &diag = *diagOr;

    ValueRef fullValueRef = valueInfo.getFullValueRef(valueRef.valueId);

    // Use a clear bitvector of the right size so we print the entire value
    // being referenced even if only part of it is missing.
    BitVector allMissing(consumedValues.size(), true);
    valueRef.markBits(allMissing, false);
    addBadValueNameToDiag(valueRef, allMissing, valueSet, diag);
    diag << " cannot be consumed, because ";

    if (valueRef.isAllPresent(consumedValues) &&
        (valueRef == fullValueRef ||
         !fullValueRef.isAllPresent(consumedValues))) {
      diag << "it";
    } else {
      // If some fields are present and others are missing, complain about the
      // first whole field that is missing.
      auto aliveValues = consumedValues;
      aliveValues.flip();
      addBadValueNameToDiag(valueRef, aliveValues, valueSet, diag);
    }
    diag << " is used later";
  }

  valueRef.markBits(consumedValues, true);

  if (!dryRun) {
    ImplicitLocOpBuilder builder(op.getLoc(), &op);

    /// Emit a debug kill marker for the value if it is tracked with debug info
    /// and if full value is destroyed (unless extending trivial debug
    /// lifetimes).
    // TODO(#34115): Emit fragment end-of-life for partial destruction.
    const ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
    if (info.debugVariable && valueRef.startBit == info.startValueBit &&
        valueRef.endBit == info.endValueBit &&
        !valueSet.shouldSuppressDebugKill(info)) {
      DebugInfo::KillOp::create(builder, info.debugVariable);
    }

    emitLifetimeEndAfter(value, &op);
  }
}

/// Check a use of a value.  Iff this is the /last/ use of the value, emit a
/// destructor of the overall value.  The 'opWithUse' value (if present)
/// indicates the operation performing the use.  This enables copy ctor elision,
/// but this is null at the start of block/function for example.
void DestructorInsertion::checkUse(Value value, bool isDeref,
                                   DestructorInserter &dtorInserter) {
  // If this is a direct reference to a value, we are tracking it, meaning
  // there are dedicated bits in the consumedValues bitvector that represent
  // the consumption state of this value.
  if (ValueRef access = valueSet.getDirectValueRef(value, isDeref)) {
    (void)scheduleNeededDtors(access, dtorInserter, value);
    return;
  }

  // We are not tracking this value directly, it could be tied to an origin
  // declared by a value we do track. If this is the case, check these values
  // for destruction.
  SmallVector<InteriorOriginAttr> interiorOrigins;
  for (ValueRef access :
       valueSet.getValueRefsForAccess(value, isDeref, interiorOrigins)) {
    // Do not pass "value" here, because it will refer to the reference, which
    // may not be to the actual tracked value for 'access'.  For example, in
    // 'use(cond ? a : b)' we want to think about "a" and "b".
    (void)scheduleNeededDtors(access, dtorInserter);
  }
}

/// This operation defines the specified value.  If the value is dead on
/// arrival, emit a destructor of the value.
void DestructorInsertion::checkDef(Value value, Operation &op, bool isDeref,
                                   DestructorInserter &dtorInserter) {
  // If there is no use of the value we are defining, scheduleNeededDtors will
  // emit a dtor after the op. This happens when we have things like:
  //
  //   init(&aggregate)
  //   ...
  //   aggregate.field1 = newValue  <<-- we are here
  if (ValueRef direct = valueSet.getDirectValueRef(value, isDeref)) {
    bool isFullUseDestroy = scheduleNeededDtors(direct, dtorInserter, value);

    // Emit a lifetime start for the value if this is a var decl. Look through
    // the same sugar as emitLifetimeEnd.
    if (!dryRun) {
      if (Value varDeclValue = stripToVarDeclLookThrough(value);
          varDeclValue.getDefiningOp<VarDeclOp>()) {
        // Emit this above the operation.
        ImplicitLocOpBuilder builder(op.getLoc(), &op);
        emitDebugInit(value, direct, builder);
        VarLifetimeStartOp::create(builder, varDeclValue);
      }
    }

    // If the destroyed value is a user-defined value that was just defined,
    // warn about the useless store.
    if (!dryRun && isFullUseDestroy) {
      ValueInfo &valueEntry = valueSet.getValueInfo(direct.valueId);

      // Don't warn about assignments into synthesized temporaries or arguments.
      if (valueEntry.shouldWarnOnUnusedAssignment()) {
        // Specialize the message for 'ref' values.
        auto varDecl = valueEntry.value.getDefiningOp<VarDeclOp>();
        if (varDecl && varDecl.getKind() == VarDeclKind::Ref) {
          // Ref's can only have a single store - their initializer. If unused,
          // then the ref is never used.
          dtorInserter.emitWarning(varDecl.getLoc(), "ref '")
              << varDecl.getName().str() << "' was never used, remove it?";
        } else {
          auto &diag = dtorInserter.emitWarning(op.getLoc(), "assignment to ");
          BitVector allMissing(consumedValues.size(), true);
          direct.markBits(allMissing, false);
          addBadValueNameToDiag(direct, allMissing, valueSet, diag);
          diag << " was never used; assign to '_' instead?";
        }
      }
    }

    direct.markBits(consumedValues, false);
    return;
  }

  // For indirect references, we treat this as a use, which will insert dtor
  // calls if this was the last use of any indirectly referenced values.
  checkUse(value, isDeref, dtorInserter);

  // Otherwise, we need to direct-emit a destructor call of the reference
  // itself since this operation will overwrite the value and we can't model
  // it in a field sensitive way.  The uninitialized checker verified that the
  // value is guaranteed live-in when nontrivial and indirect.
  if (!valueSet.isTrivial(value, isDeref) && !dryRun) {
    Type valueTy = ValueRef::getDereferencedType(value.getType(), isDeref);

    // Ok, this may run the destructor.  Make we account for any origins the
    // type captures, because the destructor may reference them.
    for (auto origin : valueSet.getOriginFinder().findOriginsIn(valueTy))
      if (!isa<AnyOriginAttr>(origin))
        checkOriginAccesses(origin, dtorInserter);

    // Destructor call goes ahead of the mutation, not after.
    ImplicitLocOpBuilder builder(op.getLoc(), &op);
    DestructorInserter beforeDtorInserter(builder, valueSet, diagsToEmit);
    beforeDtorInserter.add(value, /*Just do it*/ ValueRef(0, 0, 0, isDeref));
    beforeDtorInserter.emitDestructors();
  }
}

/// Check any unstructured origins that are accessed by the operation.
void DestructorInsertion::checkOriginAccesses(
    TypedAttr origin, DestructorInserter &dtorInserter) {
  // Iff this is the /last/ use of the value, emit a dtor for the value.
  SmallVector<InteriorOriginAttr> interiorOrigins;
  for (auto access : valueSet.getValueRefsForOrigin(origin, interiorOrigins))
    (void)scheduleNeededDtors(access, dtorInserter);
}

/// If the specified valueRef corresponds to a trivial value or subfield, clear
/// the bits associated with it in 'bits'.  This is recursive, because valueRef
/// may refer to a subfield of the overall value.
static void clearTrivialFields(ValueRef valueRef, Type valueType,
                               BitVector &bits, ValueSet &valueSet) {
  // If all the bits are already clear, we're done.
  if (valueRef.isAllMissing(bits))
    return;

  // If this value is trivial then clear the bits and we're done!
  if (valueSet.isTrivial(valueType, /*isIndirect=*/false)) {
    valueRef.markBits(bits, false);
    return;
  }

  auto valueDRType = sugarDynCast<LIT::StructType>(valueType);
  if (!valueDRType) // Trait values are not trivial.
    return;

  // Otherwise, this may be a subfield of an overall value.  Zoom in to see if
  // valueRef is referring to a trivial subfield of the overall object.
  unsigned nextBit = 0;
  for (auto field : valueSet.getTypeDeclInfo()
                        .getStructInfoForType(valueDRType)
                        .decl.getFieldDecls()) {
    // Rebound the field type first before we query the number of fields. This
    // resolves potential generic fields.
    // ```
    // struct Pair[T: Movable](Movable):
    //    var first: T
    // ```
    Type fieldType = field.getReboundType(valueDRType);
    unsigned numBits = valueSet.getTypeDeclInfo().getNumFieldsInType(fieldType);
    // If this field has consumed bits, and if has trivial type, force it
    // back to being non-consumed.  This can allow the proper correctness
    // check to work and make the error diagnostic more accurate.
    ValueRef subFieldBits = valueRef.getSubfield(nextBit, numBits);
    clearTrivialFields(subFieldBits, fieldType, bits, valueSet);
    nextBit += numBits;
  }
}

/// Given a use of a value or subfield, figure out the maximal unconsumed
/// subfield that contains it.  For example, in:
///
///   init(&aggregate)
///   use(&aggregate.field1.subfield)  <<-- We are here.
///   # Should insert: __del__(aggregate.field1)
///   init(&aggregate.field1)
///   __del__(&aggregate)
///
/// we want to return "aggregate.field1", not subfield.
static std::pair<SmallVector<StructFieldOp>, ValueRef>
computeAccessPathForMaxUnconsumedField(ValueRef use,
                                       const BitVector &consumedValues,
                                       const ValueInfo &valueInfo,
                                       TypeDeclInfo &typeDeclInfo) {
  // This only applies to indirect uses.
  if (!use.isIndirect)
    return {{}, use};

  Type valueType = cast<RefType>(valueInfo.value.getType()).getElementType();

  // Figure out where the use is WITHIN the value.
  ValueRef fullValueRef = valueInfo.getFullValueRef(use.valueId);
  unsigned numValueBits = fullValueRef.getNumBits();
  ValueRef useWithinValue = use.getWithoutBaseOffset(fullValueRef.startBit);
  unsigned totalOffset = fullValueRef.startBit;

  // Drill down into this value until we find something that isn't consumed.
  SmallVector<StructFieldOp> result;
  while (consumedValues[totalOffset + numValueBits - 1]) {
    // Okay, we must be accessing some subfield of this total value.  Figure out
    // which one, it must be field sensitive.
    auto [fieldDecl, fieldStartBit, fieldNumBits] =
        typeDeclInfo.getFieldContaining(sugarCast<LIT::StructType>(valueType),
                                        useWithinValue.startBit);

    // Don't drill into the subfield if we're spanning multiple of them.
    if (useWithinValue.getNumBits() > fieldNumBits)
      break;

    // We're drilling into this field.
    result.push_back(fieldDecl);
    useWithinValue = useWithinValue.getWithoutBaseOffset(fieldStartBit);
    totalOffset += fieldStartBit;
    numValueBits = fieldNumBits;
    valueType = fieldDecl.getType();
  }

  return {std::move(result),
          ValueRef(use.valueId, totalOffset, totalOffset + numValueBits,
                   /*isIndirect=*/true)};
}

/// There is a use of the specified 'use' portion of a live value.  If this
/// is the last use of some value, schedule a destructor to clean it up.
/// 'value' is an optional value indicating the MLIR value corresponding to
/// this, which is useful to avoid emitting redundant lit.struct.ger
/// instructions when we already have it.
///
/// Returns true if the destructor was scheduled to destroy the entire use.
bool DestructorInsertion::scheduleNeededDtors(ValueRef use,
                                              DestructorInserter &dtorInserter,
                                              Value value) {
  assert(use && "Only works on valid refs");

  // If the accessed value had an error already or nothing in this value needs
  // destroying, then ignore the request.
  ValueInfo &valueInfo = valueSet.getValueInfo(use.valueId);
  if (valueInfo.hasErrorDiagnosed || use.isAllPresent(consumedValues))
    return false;

  // If we are just computing the consumedValue set, don't actually insert any
  // destructor calls.
  if (dryRun) {
    use.markBits(consumedValues, true);
    return false;
  }

  // 'self' in an initializer is modeled as having its whole-object bit set
  // on entry to the function, but the fields may be in partially initialized
  // states throughout the body of the initializer.  We only treat the full
  // object as being initialized if all of its fields are.  This allows the
  // definition and rewrite of 'self' to work correctly, but doesn't try to
  // run the destructor on a partially initialized self.
  if (valueInfo.isFullObjectLiveOnEntry &&
      valueInfo.endValueBit == use.endBit &&
      valueInfo.startValueBit == use.startBit) {
    // If some of the fields are already missing, don't destroy self.
    --use.endBit;
    if (!use.isAllMissing(consumedValues))
      consumedValues[use.endBit] = true;
    ++use.endBit;

    // If this was the only missing bit, then we're good.
    if (use.isAllPresent(consumedValues))
      return false;
  }

  // Check to see if this whole value needs to be destroyed.
  bool isFullObjectDestroy = !consumedValues[use.endBit - 1];

  // If this is the last use of some subfield of a value that needs to be
  // destroyed, emit a destructor for the WHOLE overall value.
  //
  //   init(&aggregate)
  //   use(&aggregate.field1)
  //   use(&aggregate.field2.subelt)  <<-- We are here.
  //   # Should insert: __del__(&aggregate)
  //
  // In this case, we destroy the overall value.  However, we may be in a field
  // sensitive case where the subfield is getting reinitialized, e.g.:
  //
  //   init(&aggregate)
  //   use(&aggregate.field1.subfield)  <<-- We are here.
  //   # Should insert: __del__(aggregate.field1)
  //   init(&aggregate.field1)
  //   __del__(&aggregate)
  //
  // we have to destroy 'aggregate.field1'.  Figure out what access path we need
  // to destroy.
  auto [accessPath, adjustedUse] = computeAccessPathForMaxUnconsumedField(
      use, consumedValues, valueInfo, valueSet.getTypeDeclInfo());

  // If we were passed in a field that matches what we need, use it to avoid
  // inserting additional GER operations.  Otherwise we re-derive from the root.
  bool didAdjustUse = use != adjustedUse;
  if (!value || use != adjustedUse) {
    value = valueInfo.value;
    use = adjustedUse;

    // Drill into the right field.
    for (StructFieldOp subfield : accessPath)
      value = RefStructGEROp::create(dtorInserter.builder, value, subfield);
  }

  // If this is a store to a subfield of a non-trivial value being destroyed,
  // then the store may not be dead: it can affect the behavior of the
  // destructor. Let the caller know about this so we don't warn about dead
  // stores.
  if (didAdjustUse && !valueSet.isTrivial(value, use.isIndirect))
    isFullObjectDestroy = false;

  auto valueUseType = use.getValueType(value);

  // Get the type for the value so we can poke at it.
  // If a generic type or trivial, then emit a destructor call (or nothing).
  auto valueType = sugarDynCast<LIT::StructType>(valueUseType);
  if (!valueType) {
    // We are going to emit a destructor for the specified ValueRef, so all none
    // of the things we are about to destroy should already be destroyed.
    assert(use.isAllMissing(consumedValues) &&
           "cannot have partially consumed object");
    dtorInserter.add(value, use);
    use.markBits(consumedValues, true);
    return isFullObjectDestroy; // Destroyed the full value.
  }

  // The type we are destroying may reference other values, e.g. consider a
  // InterestingPointer[origin o].  Its destructor could touch the origin, so we
  // need to make sure to treat any destructions of the interesting pointer as a
  // use of the origins it may reference.
  for (auto origin : valueSet.getOriginFinder().findOriginsIn(valueUseType))
    checkOriginAccesses(origin, dtorInserter);

  // Trivial types don't have __del__ methods and can't be tracked, so if
  // this is referring to one of them, make sure to clear the bits so we
  // don't think they need to be destroyed.
  clearTrivialFields(use, valueType, consumedValues, valueSet);

  // If we need to destroy the whole value, we can just use an empty BitVector,
  // otherwise we need to specify which subelements are to be destroyed, so we
  // copy it.
  BitVector fieldsToDestroy;
  if (!use.isAllMissing(consumedValues))
    fieldsToDestroy = consumedValues;
  dtorInserter.add(value, use, std::move(fieldsToDestroy));
  use.markBits(consumedValues, true);

  // Return true if we destroyed the full reference.
  return isFullObjectDestroy;
}

void DestructorInsertion::emitDebugInit(Value value, ValueRef valueRef,
                                        ImplicitLocOpBuilder &builder) {
  assert(!dryRun && "shouldn't be called in a dry run");
  ValueInfo &info = valueSet.getValueInfo(valueRef.valueId);
  // Insert debug value if full value is initialized.
  if (info.debugVariable && valueRef.startBit == info.startValueBit &&
      valueRef.endBit == info.endValueBit) {
    // The IR type needs to be deref'ed to get the source type. Encode the IR
    // type as a pointer type.
    Type irType = value.getType();
    auto newIrValue = DebugInfo::DIIRValueExprAttr::get(irType);
    Type derefTy = irType;
    if (auto refType = dyn_cast<RefType>(irType))
      derefTy = refType.getElementType();
    DebugInfo::DIExprAttr conversion =
        DebugInfo::DIDerefExprAttr::get(newIrValue, derefTy);

    // For ref bindings (e.g. loop variables), the VarDecl element is itself a
    // ref. Add an extra deref to reach the actual value through the inner ref.
    auto varDecl = info.value.getDefiningOp<VarDeclOp>();
    if (varDecl && varDecl.getKind() == VarDeclKind::Ref) {
      if (isa<RefType>(varDecl.getType().getElementType())) {
        Type innerTy = derefTy;
        if (auto innerRef = dyn_cast<RefType>(derefTy))
          innerTy = innerRef.getElementType();
        conversion = DebugInfo::DIDerefExprAttr::get(conversion, innerTy);
      }
    }

    DebugInfo::ValueOp::create(builder, value, info.debugVariable, conversion);
  }
}

/// Insert destructors calls into the start of 'block' for objects in the
/// 'fullSetToDestroy' that are not already in the 'currentConsumeSet'.  This is
/// used at control flow merges.
///
/// This does not modify 'consumedValues', and does respect 'dryRun'.
void DestructorInsertion::destroyValuesAtEntryIfNeeded(
    const BitVector &currentConsumeSet, Block &block,
    const BitVector &fullSetToDestroy, Location loc) {
  // If we are in a dry run or the two sets match, or the block is unreachable,
  // don't actually insert anything.
  if (dryRun || currentConsumeSet == fullSetToDestroy ||
      // Don't do anything if the entry to this block is unreachable. This will
      // be false for reachable blocks and blocks that contain no-return calls.
      !currentConsumeSet[0])
    return;

  // entriesToDestroy = fullSetToDestroy & ~currentConsumeSet.
  BitVector entriesToDestroy = fullSetToDestroy;
  entriesToDestroy.reset(currentConsumeSet);

  // Move consumedValues out of the way so we don't break it.  We need to use
  // scheduleNeededDtors below, which is hard coded to mutate consumedValues.
  BitVector savedConsumedValues = std::move(consumedValues);

  // We *only* want to destroy the values in entries, not any other values that
  // may be partially overlapped, so mark all the other things as "already
  // destroyed".  This is to work with 'scheduleNeededDtors'.
  assert(&entriesToDestroy != &consumedValues &&
         "This logic doesn't work when passed 'consumedValues' directly");
  consumedValues = entriesToDestroy;
  consumedValues.flip();

  // As we scan through bits, we walk through corresponding ValueInfos to know
  // what we are working with.
  MutableArrayRef<ValueInfo> valueInfos = valueSet.getValueInfos();
  size_t nextValueInfo = 0;

  // Any dtor calls will be emitted at the start of the block.
  DestructorInserter dtorInserter(
      ImplicitLocOpBuilder(loc, &block, block.begin()), valueSet, diagsToEmit);

  int nextToDestroy = entriesToDestroy.find_first();
  while (nextToDestroy != -1) {
    // Figure out which valueInfo this is.
    while (!valueInfos[nextValueInfo].contains(nextToDestroy)) {
      ++nextValueInfo;
      assert(nextValueInfo != valueInfos.size() &&
             "nothing contains this bit?");
    }

    // Ok, we know that we are destroying some field of this value, find the
    // whole value so we know the MLIR value.
    ValueRef fullValueRef = valueSet.getFullValueRef(nextValueInfo);

    // Emit destructor calls for the entire value or the correct subfields that
    // need to be destroyed.
    (void)scheduleNeededDtors(fullValueRef, dtorInserter);

    // Find the next object to destroy.
    nextToDestroy = entriesToDestroy.find_next(fullValueRef.endBit - 1);
  }

  dtorInserter.emitDestructors();

  // Restore consumedValues.
  consumedValues = std::move(savedConsumedValues);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CHECKLIFETIMES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

struct CheckLifetimes : impl::CheckLifetimesBase<CheckLifetimes> {
  using CheckLifetimesBase::CheckLifetimesBase;

  void runOnOperation() override {
    // Build the shared, immutable module-level state.
    std::vector<FnOp> functionVector;
    WholeProgramState sharedState(getOperation(), functionVector);

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    mlir::LockedSymbolTableCollection sharedSymtabs(analysis.getSymbolTables());

    TypeDeclInfo typeDeclInfo(&sharedState, getOperation(), &sharedSymtabs);

    std::atomic<bool> hadError = false;
    // Each worker thread gets its own Cache (TypeDeclInfo + CachedOriginFinder)
    // that persists across all functions it processes; it is reset between
    // functions.
    PerThreadCache cache(std::move(typeDeclInfo));
    // `functionVector` holds only top-level functions. Each task walks its
    // function's nested functions and processes the whole subtree on one
    // thread. A nested function (an old closure) reads SSA values from the
    // function it is nested in, so keeping the subtree on a single thread
    // avoids racing that function's own IR mutation (destructor insertion);
    // distinct top-level functions are op-disjoint.
    //
    // FIXME(MOCO-3942): remove this code once old closures are removed -- every
    // function is then op-disjoint and this becomes a plain per-function loop.
    M::parallelForEach(
        &getContext(), functionVector,
        [&](PerThreadCache &cache, FnOp topFunc) {
          topFunc.walk([&](FnOp func) {
            if (failed(processFunction(func, cache)))
              hadError = true;
            cache.reset();
          });
        },
        cache);

    if (hadError)
      signalPassFailure();
  }

  LogicalResult processFunction(FnOp func, PerThreadCache &perThreadCache);
};
} // namespace

LogicalResult CheckLifetimes::processFunction(FnOp func,
                                              PerThreadCache &perThreadCache) {

  // If the function is a trait function or something else unreachable, we don't
  // need to process it.
  Block &funcBody = func.getBodyRegion().front();
  if (isa<UnreachableOp>(funcBody.front()))
    return success();

  // Similarly, if this is a signature resolved function (not body resolved),
  // then ignore it.
  if (auto endFunc = dyn_cast<EndFnOp>(funcBody.front()))
    if (endFunc.getUnresolved())
      return success();

  // Walk #1: Collect all of the values declared in the function that have
  // ownership to track, and number them.
  ValueSet valueSet(perThreadCache, func, extendTrivialDebugLifetimes);

  // Walk #2: Scan the function and identify any uses of values that are not
  // defined, emitting diagnostics as we go.
  bool hadErrors = false;
  {
    InteriorOriginTracker interiorOriginTracker(perThreadCache, func);
    // interiorOriginTracker.dump();
    UninitializedValueScan(valueSet, interiorOriginTracker).scanFunction(func);
    hadErrors |= interiorOriginTracker.hadAnyErrors();
  }

  // Walk #3: Scan the function bottom-up, inserting destructor calls, inserting
  // lifetime markers, and eliding temporaries.
  DestructorInsertion(valueSet).scanFunction();

  // Now that we've transformed the function, look for any vardecls that only
  // have lifetime markers.  They can be removed, because all their uses got
  // forwarded or rewritten.
  for (ValueInfo &info : valueSet.getValueInfos()) {
    if (!info.value) // Already removed value.
      continue;

    auto varDecl = info.value.getDefiningOp<VarDeclOp>();
    if (!varDecl)
      continue;

    if (!info.isEverUsed && varDecl.shouldWarnAboutUnused()) {
      mlir::emitWarning(varDecl.getLoc())
          << (varDecl.getKind() != VarDeclKind::Ref ? "variable '" : "ref '")
          << varDecl.getName().str() << "' was never used, remove it?";
    }

    // Check to see if there are any uses other than lifetime markers.
    bool hasInterestingUse = false;
    for (Operation *user : varDecl->getUsers()) {
      if (isa<VarLifetimeStartOp, VarLifetimeEndOp, RebindOp, RefImmutOp,
              RefUpcastOp>(user) &&
          user->use_empty())
        continue;

      hasInterestingUse = true;
      break;
    }
    if (hasInterestingUse)
      continue;

    // Okay, nothing interesting happening here.  Remove any lifetime markers
    // and remove the vardecl as well.
    while (!varDecl->use_empty())
      varDecl->user_begin()->erase();
    varDecl->erase();
  }

  // Return failure if we generated errors for any of the tracked values.
  hadErrors |= llvm::any_of(valueSet.getValueInfos(), [&](ValueInfo &info) {
    return info.hasErrorDiagnosed;
  });
  return failure(hadErrors);
}
