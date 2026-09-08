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
// This file defines logic that reasons about value and memory object origins:
// what an operation defines and consumes.
//
//===----------------------------------------------------------------------===//

#include "Mojo/LITDialect/OriginTrackable.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Return true if the specified MLIR type is obviously a trivial register type.
static bool isTypeObviouslyTrivial(Type type) {
  return isa<KGEN::NoneType, IntegerType, RefType>(type);
}

/// Collect interior origins defined by `userResultType` into
/// `interiorOriginsDefined`. Subtree origins (~a) are modeled as a synthetic
/// interior origin for the same analysis.
static void collectInteriorOriginsDefinedByType(
    Type userResultType, CachedOriginFinder &originFinder,
    SmallVectorImpl<InteriorOriginAttr> &interiorOriginsDefined) {
  for (TypedAttr origin : originFinder.findOriginsIn({userResultType})) {
    // Given a def of something with an "a.list["x"].second.field["y"].z"
    // origin, we need to mark the "x" and "y" interior origins live.
    origin.walk([&](Attribute nested) {
      if (auto into = dyn_cast<InteriorOriginAttr>(nested)) {
        interiorOriginsDefined.push_back(into);
      } else if (auto subtree = dyn_cast<OriginSubtreeAttr>(nested))
        interiorOriginsDefined.push_back(getInteriorForSubtreeOrigin(subtree));
    });
  }
}

//===----------------------------------------------------------------------===//
// OriginTrackable
//===----------------------------------------------------------------------===//

OriginTrackable::OriginTrackable(Value v) {
  if (!v) // Null value isn't tracked.
    return;

  // VarDeclOp is uninit and ends that way.
  if (auto varDecl = v.getDefiningOp<VarDeclOp>()) {
    // Implicit temporaries get names like "*anonymous", don't print that!
    if (varDecl.getKind() == VarDeclKind::Synthesized)
      name = StringAttr::get(v.getContext(), "(expression temporary)");
    else
      name = varDecl.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endInitState = EndsUninit;

    // If this is a vardecl shadow of a register passable 'out' argument, then
    // the value is treated as if its whole-object bit is live on entry.  This
    // allows it to be fieldwise assigned.
    if (varDecl.getKind() == VarDeclKind::InitOutArg)
      isFullObjectLiveOnEntry = true;
    return;
  }

  if (v.getDefiningOp<LoadConsumeOp>() ||
      v.getDefiningOp<ParamMaterializeOp>()) {
    name = StringAttr::get(v.getContext(), "(anonymous value)");
    isIndirect = false;
    startsUninit = true;
    endInitState = EndsUninit;
    return;
  }

  // The lit.ref.from_pointer op takes an origin-tracked reference.  We
  // unconditionally model this as same liveness on entry to the function as on
  // exit, because some control flow paths may never execute the operation.
  //
  // When the op is executed to take ownership of the raw pointer,
  // CheckLifetimes will notice its actual effect: if it is init on entry and
  // uninit on exit, CheckLifetimes will ensure the value gets consumed or
  // destroyed.
  if (auto refFromPtr = v.getDefiningOp<RefFromPointerOp>()) {
    name = StringAttr::get(v.getContext(), "(reference value)");
    isIndirect = true;
    startsUninit = refFromPtr.getEndUninit();
    endInitState = startsUninit ? EndsUninit : EndsInit;
    return;
  }

  // This is a horrible hack for the REPL. :-(
  if (auto refFromPtr = v.getDefiningOp<RefFromPointerREPLOp>()) {
    name = refFromPtr.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;
    return;
  }

  /// Owned results of function calls are tracked as being initialized when
  /// defined but needing to be destroyed by the end of function.
  if (OpResult res = dyn_cast<OpResult>(v)) {
    // We have several different common cases, including:
    // 1) a trivial MLIR type that doesn't matter, or a register passable
    //    trivial type that we can track it for convenience.
    // 2) an non-trivial register passable (e.g. Arc) which we need to track
    //    as being defined by the call and needing to be consumed before the
    //    function exit.
    // 3) a ref-result type, which is tracked by origin tracking but isn't
    //    owned.  We don't (and can't) track this, but CheckLifetimes will
    //    notice the origin it contains.

    // If this is a ref result (#3) or a raw !lit.ref returned with
    // __mlir_type, we *can't* track it as an owned result, because this
    // doesn't own the value!
    if (isTypeObviouslyTrivial(v.getType()))
      return;

    if (isa<KGENCallOpInterface, LIT::CallIndirectOp>(res.getOwner())) {
      // Otherwise we tell CheckLifetimes to track it because it is either an
      // owned register result or it doesn't matter because it is trivial.
      name = StringAttr::get(v.getContext(), "(call result)");
      isIndirect = false;
      startsUninit = true;
      endInitState = EndsUninit;
      return;
    }

    // The results of an if operation can be owned.
    if (isa<HLCF::IfOp, HLCF::ElifOp, ParamIfOp>(res.getOwner())) {
      name = StringAttr::get(v.getContext(), "(if result)");
      isIndirect = false;
      startsUninit = true;
      endInitState = EndsUninit;
      return;
    }

    // Otherwise, some other unknown op result.
    return;
  }

  // If this is a function argument, check to see what ownership it has.
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg || !bbArg.getOwner())
    return;
  FnTypeGeneratorType signature;
  bool isInit = false;
  if (auto func = dyn_cast<FnOp>(bbArg.getOwner()->getParentOp())) {
    signature = func.getFuncTypeGenerator();
    isInit = func.getSpecialFunctionInfo().isInitializer();
  }
  if (!signature)
    return;

  unsigned argIdx = bbArg.getArgNumber();
  switch (signature.getArgConvention(argIdx)) {
  case ArgConvention::ImmReg:
    // This is immutable so don't need to be tracked.
    return;

  case ArgConvention::ImmMem:
  case ArgConvention::Mut:
  case ArgConvention::MutRef:
  case ArgConvention::Ref:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsInit;
    break;

  case ArgConvention::OwnedReg:
    isIndirect = false;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    isIndirect = true;
    startsUninit = false;
    endInitState = EndsUninit;
    break;
  case ArgConvention::ByRefResult:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnNormal;

    // Initializers allow member-wise initialization of 'self' to construct a
    // full value.
    if (isInit)
      isFullObjectLiveOnEntry = true;
    break;
  case ArgConvention::ByRefError:
    isIndirect = true;
    startsUninit = true;
    endInitState = InitOnError;
    break;
  }

  name = signature.getArgName(argIdx);
  if (name.empty()) {
    name = StringAttr::get(v.getContext(), "(positional-only argument # " +
                                               Twine(argIdx) + ")");
  }
}

/// This constructor checks to see if the value is trackable or a field of a
/// trackable.  If so it identifies the underlying object being referenced. If
/// not, this returns a null value.
Value OriginTrackable::findUnderlyingValueFromField(Value value) {
  // If there are any GEP operations into the struct, dig through them.
  bool hadGEP = false;

  while (true) {
    if (auto structGER = value.getDefiningOp<RefStructGEROp>()) {
      hadGEP = true;
      value = structGER.getContainer();
    } else if (auto rebindOp = value.getDefiningOp<RebindOp>()) {
      value = rebindOp.getOperand();
    } else if (auto immut = value.getDefiningOp<RefImmutOp>()) {
      value = immut.getOperand();
    } else if (auto upcast = value.getDefiningOp<RefUpcastOp>()) {
      value = upcast.getOperand();
    } else {
      break;
    }
  }

  // Check if there is a base value.
  OriginTrackable result(value);
  // If we had a GEP of this value but it doesn't have indirect storage, then
  // we aren't actually tracking the pointers off this value field sensitively,
  // so we can't be confident about what is going on with it.
  if (!result || (hadGEP && !result.isIndirect))
    return Value();
  // Otherwise, use whatever we found.
  return value;
}

//===----------------------------------------------------------------------===//
// OperationValueEffects
//===----------------------------------------------------------------------===//

/// This is a helper for `OperationEffects::analyze` split out since calls are
/// so interesting.
void OperationEffects::analyzeCallOp(Operation &op) {
  FuncType signature;
  OperandRange callArguments = op.getOperands();
  ArrayRef<ArgConvention> conventions;

  if (auto directCall = dyn_cast<KGENCallOpInterface>(op)) {
    // These all have the callee as a parameter, not operand.
    signature = directCall.getCalleeType().getBody();
    conventions = signature.getArgConventions().drop_back(
        signature.getNumAsyncReturnSlots());

    // CreateClosureOp has a subset of the operands of a call.
    if (isa<CreateClosureOp>(op))
      conventions = conventions.take_front(op.getNumOperands());

    assert(conventions.size() == op.getNumOperands());
  } else {
    // Otherwise must be a call indirect.
    signature = cast<LIT::CallIndirectOp>(op).getCallee().getType().getBody();
    conventions = signature.getArgConventions();

    // We use the callee value, and process the rest as operands.
    operands.push_back({op.getOperand(0), OperandEffect::regUse});
    assert(conventions.size() == op.getNumOperands() - 1);
    callArguments = callArguments.drop_front();
  }

  // Callees marked `@__unsafe_nested_origins_read_only` promise to
  // only read origins reachable through their arguments.  Model any
  // type-embedded origins as readonly-only so CheckLifetimes extends
  // lifetimes without treating them as exclusive mutable accesses.
  bool originsAccessesAreReadOnly = signature.getIsNestedOriginsReadOnly();

  /// Argument conventions cause a direct use of the register of pointee, and
  /// handling them specifically allows us to be field sensitive in cases where
  /// the access is directly attributable to a Value.
  auto getOperandEffectForConvention = [&](ArgConvention conv,
                                           Type argType) -> OperandEffect {
    switch (conv) {
    case ArgConvention::OwnedReg:
      return OperandEffect::regConsume;
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      return OperandEffect::memConsume;
    case ArgConvention::ImmReg:
      return OperandEffect::regUse;
    case ArgConvention::ImmMem:
      return OperandEffect::memLoad;
    case ArgConvention::Mut:
    case ArgConvention::MutRef:
      return OperandEffect::memMut;
    case ArgConvention::Ref: {
      // If the reference is guaranteed immutable, then treat it as a load,
      // otherwise it might be a mutation. If originAccessesAreReadOnly is true,
      // then we this "ref" argument is treated as read only.
      bool isImmut = originsAccessesAreReadOnly ||
                     cast<RefType>(argType).isMutableKnown(false);
      return isImmut ? OperandEffect::memLoad : OperandEffect::memMut;
    }
    case ArgConvention::ByRefError:
    case ArgConvention::ByRefResult:
      return OperandEffect::memStoreOwned;
    }
    llvm_unreachable("invalid input convention");
  };

  auto addArgument = [&](Value arg, ArgConvention conv,
                         bool noIndirect = false) {
    // Get normal argument effect.
    Type argType = arg.getType();
    auto effect = getOperandEffectForConvention(conv, argType);

    // If this is a normal register use, and if the value is a reference
    // (whether the argument convention is fancy or if it is an explicitly
    // passed reference) treat this as a field sensitive access so we can
    operands.push_back({arg, effect});

    // If the caller doesn't want us to add type-based origin effects, don't.
    if (noIndirect)
      return;

    // Do not add type-based origin effects for result arguments.  They are
    // returned, not accessed and therefore don't conflict with the inputs.
    if (conv == ArgConvention::ByRefResult || conv == ArgConvention::ByRefError)
      return;

    // If this is a memConsume or memStoreOwned, then the origin of the
    // reference is handled directly, strip it off.  Otherwise handle read,
    // mut, etc operands as just any-old reference use.
    argType = RefType::stripRefConvention(argType, conv);

    // In addition to the direct (field-sensitive) effect of loading/storing
    // the bits, the callee may do whatever it wants with origins embedded
    // in the type.
    for (TypedAttr origin : originFinder.findOriginsIn({argType})) {
      if (originsAccessesAreReadOnly)
        origin = OriginMutCastAttr::get(origin, false);
      origins.push_back({origin, arg});
    }
  };

  // Find the actual result type of the call.  We can't pick this up from the
  // signature type with getUserResultType because that won't have implicit
  // origins substituted in.
  // TODO(remove implicit origins)!
  assert(op.getNumResults() == 1);
  Type userResultType = op.getResult(0).getType();

  for (auto [idx, arg, convention] :
       llvm::enumerate(callArguments, conventions)) {

    if (convention == ArgConvention::ByRefResult)
      userResultType = cast<RefType>(arg.getType()).getElementType();

    // If this is VariadicList/VariadicPack, dig out the original arguments so
    // we can model owned arguments correctly. This provides two benefits:
    //   1) given "direct" access information, it allows us to model 'owned'
    //      argument conventions which consume the operand, something origin
    //      accesses cannot model (because it requires field sensitivity).
    //   2) it allows us to reason about the varargs uses field-sensitively,
    //      e.g. you can pass `a.x` through varargs and `a.y` through a 'mut'
    //      without the compiler imagining a conflict on "a" just like other
    //      arguments.
    // TODO: It would be nice to handle more fine grain effects in a general way
    // on calls.  This is a hack.
    // TODO(field-sensitive origins): remove this hack.
    // TODO: This should be removed. This is disabled for packs passed by-ref
    // when they are owned.
    if ((signature.isPack(idx) || signature.isPosVarArg(idx)) &&
        // Thunks can directly forward pack arguments, but we don't need to
        // model them.
        !isa<BlockArgument>(arg)) {

      ArgConvention argConvention = signature.getVariadicConvention(idx);

      // Find all the references getting passed into the list/pack so we can
      // process them individually.
      TypedAttr extraOrigin;
      for (auto elt :
           OriginTrackable::decodeIndividualVariadicArguments(arg, extraOrigin))
        addArgument(elt, argConvention);

      // This hack makes sure we extend the lifetime of the VarDecl containing
      // the variadic list. VariadicList itself should capture the origin.
      if (extraOrigin)
        origins.push_back({extraOrigin, arg});

      // Also add the list/pack itself so the VariadicList/VariadicPack
      // doesn't get destroyed too early.  We already handled all the
      // individual elements, so don't redundantly process them.  Doing so is
      // a problem for owned operands.
      addArgument(arg, convention, /*noIndirect=*/true);
      continue;
    }

    addArgument(arg, convention);
  }

  // The callee may also access origins from its capture set.
  for (TypedAttr origin :
       originFinder.findOriginsIn({}, signature.getCaptureOrigins())) {
    if (originsAccessesAreReadOnly)
      origin = OriginMutCastAttr::get(origin, false);
    origins.push_back({origin, Value()});
  }

  // If the result is defining an owned register value, then we treat this as
  // a definition
  results.push_back(isTypeObviouslyTrivial(op.getResult(0).getType())
                        ? ResultEffect::ignore
                        : ResultEffect::regDefine);

  // If the result of this function is defining an interior origin, then we
  // need to mark the interior origins live.
  if (signature.getDefinesInteriorOrigins())
    collectInteriorOriginsDefinedByType(userResultType, originFinder,
                                        interiorOriginsDefined);
}

/// This computes the effects that an operation has on any operands and result
/// values. This information is used by both phases of CheckLifetimes.
OverallOpValueEffect OperationEffects::analyze(Operation &op) {
  // Get ready to populate the new fields.
  operands.clear();
  results.clear();
  origins.clear();
  interiorOriginsDefined.clear();

  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.  These indexing operations are handled specially.
  if (isa<RefStructGEROp, RebindOp, RefImmutOp>(op) ||
      llvm::isa_and_nonnull<DebugInfo::DebugInfoDialect>(op.getDialect())) {
    if (op.getNumResults() == 1)
      results.push_back(ResultEffect::ignore);
    return {};
  }

  /// When all of the operands of an instruction have an effect and they are
  /// in a fixed order, this helper can help specify them.
  auto setOperandEffects = [&](ArrayRef<OperandEffect> operandEffects) {
    assert(operandEffects.size() == op.getNumOperands() &&
           "operand count mismatch");
    for (auto [operand, effect] : llvm::zip(op.getOperands(), operandEffects))
      operands.push_back({operand, effect});
  };

  // RefStore consumes its operand and transfers it into the result.
  if (isa<RefStoreOp>(op)) {
    setOperandEffects(
        {OperandEffect::regConsume, OperandEffect::memStoreOwned});
    return {};
  }

  // Memcpy consumes its operand and transfers it into the result.
  if (isa<MemcpyOp>(op)) {
    setOperandEffects({OperandEffect::memLoad, OperandEffect::memStoreOwned});
    return {};
  }

  // MaterializeInto overwrites the memory using the (materialized) parameter
  // value.
  if (isa<MaterializeIntoOp>(op)) {
    setOperandEffects({OperandEffect::memStoreOwned});
    return {};
  }

  // A load is a use of whatever fields are being referenced.  If this is
  // the /last/ use of a value, emit a destructor of that value.  LoadOps
  // are used to model a /borrow/ of the underlying value, so they don't
  // define a new value.
  if (auto load = dyn_cast<RefLoadOp>(op)) {
    operands.push_back({load.getOperand(), OperandEffect::memLoad});
    results.push_back(ResultEffect::ignore);
    return {};
  }
  if (auto load = dyn_cast<LoadConsumeOp>(op)) {
    operands.push_back({load.getOperand(), OperandEffect::memConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  if (auto use = dyn_cast<OwnershipUseOp>(op)) {
    auto operandEffect = isa<RefType>(op.getOperand(0).getType())
                             ? OperandEffect::memLoad
                             : OperandEffect::regUse;
    operands.push_back({use.getOperand(), operandEffect});
    return {};
  }

  // These ops consume their operands, struct.create and param.materialize
  // define a result.
  if (isa<ParamMaterializeOp>(op)) {
    for (Value o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});
    results.push_back(ResultEffect::regDefine);
    return {};
  }

  // RefFromPointerOp creates a new origin tracked value.  The
  // 'startsUninit' field impacts the execution of the operation (now), not
  // its modeling at start of the function.  We have to assume its liveness
  // at start of function is the same as its liveness at end of function
  // because not all control flow paths will execute the operation.
  if (auto refFromPtr = dyn_cast<RefFromPointerOp>(op)) {
    // Ignore the pointer input.
    ResultEffect resultEffect;
    if (refFromPtr.getStartUninit()) {
      resultEffect = refFromPtr.getEndUninit()
                         ? ResultEffect::memDefineUninitToUninit
                         : ResultEffect::memDefineUninitToInit;
    } else {
      resultEffect = refFromPtr.getEndUninit()
                         ? ResultEffect::memDefineInitToUninit
                         : ResultEffect::memDefineInitToInit;
    }
    results.push_back(resultEffect);
    return {};
  }

  // If this is a call, investigate each of the operands along with the
  // argument convention effects.
  if (isa<LIT::CallIndirectOp, KGENCallOpInterface>(op)) {
    analyzeCallOp(op);
    return {};
  }
  // A return consumes all the live-out values from the function.
  if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp, KGEN::UnreachableOp,
          HLCF::YieldOp>(op)) {
    // We always consume the result register - even if it is often trivial.
    for (auto o : op.getOperands())
      operands.push_back({o, OperandEffect::regConsume});

    // Yield doesn't need any special processing, just handling of its operands.
    if (isa<HLCF::YieldOp>(op))
      return {};

    return OverallOpValueEffect::terminatorOp;
  }

  if (auto mark = dyn_cast<OwnershipMarkInitializedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memStoreOwned});
    return {};
  }

  if (auto mark = dyn_cast<OwnershipMarkDestroyedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memMarkDestroyed});
    return {};
  }

  if (auto mark = dyn_cast<OwnershipMarkConsumedOp>(op)) {
    operands.push_back({mark.getOperand(), OperandEffect::memConsume});
    return {};
  }

  // Debuginfo ops may reference values that aren't fully initialized, so we
  // skip over them.  These indexing operations are handled specially.
  if (auto upcast = dyn_cast<RefUpcastOp>(op)) {
    results.push_back(ResultEffect::ignore);

    // RefUpcastOp can define a subtree origin.  If so, notice this and treat it
    // as a read of the source, which will validate that the source is alive.
    // Otherwise it is a noop.
    collectInteriorOriginsDefinedByType(upcast.getType(), originFinder,
                                        interiorOriginsDefined);
    if (!interiorOriginsDefined.empty())
      operands.push_back({upcast.getOperand(), OperandEffect::memLoad});
    return {};
  }

  // Local control flow ops.
  if (isa<HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, ParamForBreakOp,
          ParamForContinueOp>(op))
    return OverallOpValueEffect::localControlFlowOp;

  // If-like operations.
  if (isa<ParamIfOp, HLCF::IfOp>(op)) {
    // If-like ops can return owned results.
    for (auto opResult : op.getResults()) {
      auto resultEffect = isTypeObviouslyTrivial(opResult.getType())
                              ? ResultEffect::ignore
                              : ResultEffect::regDefine;
      results.push_back(resultEffect);
    }
    return OverallOpValueEffect::ifLikeOp;
  }

  if (isa<HLCF::ElifOp>(op)) {
    // If-like ops can return owned results.
    for (auto opResult : op.getResults()) {
      auto resultEffect = isTypeObviouslyTrivial(opResult.getType())
                              ? ResultEffect::ignore
                              : ResultEffect::regDefine;
      results.push_back(resultEffect);
    }
    return OverallOpValueEffect::elifOp;
  }

  /// This is HLCF::LoopOp.
  if (isa<HLCF::LoopOp, ParamForOp>(op))
    return OverallOpValueEffect::loopOp;

  if (isa<LIT::TryOp>(op))
    return OverallOpValueEffect::tryOp;

  assert(!isa<HLCF::SwitchOp>(op) && "Only created by LowerSuspension Points");
  return OverallOpValueEffect::unknownOp;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Find the single !lit.ref.store that stores to the specified var decl.
static Value findSingleStoreToVarDecl(VarDeclOp varDecl) {
  Value foundValue;
  for (Operation *user : varDecl.getResult().getUsers()) {
    if (auto refStore = ::dyn_cast<RefStoreOp>(user)) {
      if (refStore.getDest() == varDecl.getResult()) {
        assert(!foundValue && "expected to find a single store to a var decl");
        foundValue = refStore.getValue();
      }
    }
  }

  assert(foundValue && "expected to find a single store to a var decl");
  return foundValue;
}

/// Given an argument to a function that takes a VariadicList/VariadicPack
/// argument, look through the local constructor to the value that formed it.
/// Returns null when the argument is a forward of an already-constructed
/// container.
static Value findArgPassedToVariadicConstructor(Value val) {
  // Strip off sugar casts, mutability casts etc.
  val = RefImmutOp::stripRebinds(val);

  // This code grovels through the IR, looking for the standard pattern of:
  //
  //   %1 = lit.ref.pack.create(...)
  //   %anonymous2A_0 = lit.var.decl "anonymous*"
  //   lit.call VariadicPack::__init__(%anonymous2A_0, %1, ...)
  //   %4 = lit.load.consume / lit.ref.load %anonymous2A_0  <<= we are here.
  //
  // This happens because we're passing the VariadicPack to the callee, and
  // it has a memory-style init.
  Value loadOperand;

  // VariadicPack is a RegisterPassable type so it often is immediately
  // available.  However, it gets passed by-ref to function calls.
  // If the operand is already a reference to a pack, then use it.  Otherwise
  // we must have a register pack.  Figure out how it is formed.
  if (::isa<RefType>(getCanonicalType(val.getType()))) {
    loadOperand = RefImmutOp::stripRebinds(val);
  } else {
    if (auto load = val.getDefiningOp<RefLoadOp>())
      loadOperand = load.getOperand();
    else if (auto load = val.getDefiningOp<LoadConsumeOp>())
      loadOperand = load.getOperand();
    else
      return {};
  }
  loadOperand = RefImmutOp::stripRebinds(loadOperand);

  auto varDecl = loadOperand.getDefiningOp<VarDeclOp>();
  if (!varDecl)
    return {};

  auto storedValue = findSingleStoreToVarDecl(varDecl);

  // Look through VariadicList/VariadicPack::__init__ so we can check the
  // individual arguments for exclusivity. Without this, `mutate_pack(x, x)`
  // would not be diagnosed as an aliasing violation.
  auto call = storedValue.getDefiningOp<LIT::CallOp>();
  if (!call)
    return {};
  auto callee = call.getDirectCallee();
  if (!callee || !callee.getLeafReference().strref().starts_with("__init__"))
    return {};

  // Make sure any change to the API forces this code to get updated.
  assert(call.getNumOperands() == 1 &&
         "VariadicList/VariadicPack ctor take a single argument");
  return RefImmutOp::stripRebinds(call.getOperand(0));
}

/// Return the immutable origin parameter of a VariadicPack or VariadicList.
static TypedAttr getVariadicContainerOrigin(Value value) {
  Type type = getCanonicalType(value.getType());
  if (auto refType = dyn_cast<RefType>(type))
    type = getCanonicalType(refType.getElementType());

  auto structType = sugarDynCast<LIT::StructType>(type);
  if (!structType)
    return {};
  StringRef name = structType.getName().getValue();
  if (name != "VariadicPack" && name != "VariadicList")
    return {};
  assert(structType.getParamValues().size() > 1 &&
         isa<OriginType>(structType.getParamValues()[1].getType()) &&
         "expected a VariadicPack/VariadicList origin parameter");
  return OriginMutCastAttr::get(structType.getParamValues()[1], false);
}

/// Given the argument passed to a variadic argument, dig out the trackable
/// values passed to the VariadicList/VariadicPack constructor.  Each of these
/// may be used or consumed (by an owning variadic) so CheckLifetimes needs
/// understand the impact on the individual arguments.
///
/// This returns empty for forwarded containers, because there are no
/// individual values to track.
SmallVector<Value>
OriginTrackable::decodeIndividualVariadicArguments(Value callArgVal,
                                                   TypedAttr &extraOrigin) {
  Value ctorArg = findArgPassedToVariadicConstructor(callArgVal);

  SmallVector<Value> result;
  if (auto pack =
          ctorArg ? ctorArg.getDefiningOp<RefPackCreateOp>() : nullptr) {
    // Locally constructed VariadicPack. Each operand is an argument reference
    // rebound onto a common origin union. Strip the rebind so each keeps its
    // original origin.
    for (auto elt : pack.getOperands())
      result.push_back(RefImmutOp::stripRebinds(elt));
    return result;
  }

  Value listStorage = ctorArg;
  if (auto refLoad = ctorArg ? ctorArg.getDefiningOp<RefLoadOp>() : nullptr)
    listStorage = RefImmutOp::stripRebinds(refLoad.getOperand());

  if (auto varDecl =
          listStorage ? listStorage.getDefiningOp<VarDeclOp>() : nullptr) {
    if (auto arrayCreate = findSingleStoreToVarDecl(varDecl)
                               .getDefiningOp<POP::ArrayCreateOp>()) {
      // Locally constructed VariadicList. The parser emits:
      //   %__passed_varargs__ = lit.var.decl: !lit.ref<array<4, eltref>>
      //   %arr = pop.array.create [%1, %2, %3]
      //   lit.ref.store %arr, %__passed_varargs__
      //   lit.call VariadicList::@"__init__"(%__passed_varargs__)
      // Keep the VarDecl alive across the call. VariadicList should capture
      // this origin itself; this is a workaround because the VarDecl does not
      // exist at ParamInf time. The callee reads the VarDecl but doesn't
      // mutate it.
      extraOrigin =
          OriginMutCastAttr::get(varDecl.getType().getOrigin(), false);
      for (auto value : arrayCreate.getOperands())
        result.push_back(value);
      return result;
    }
  }

  // Forwarded list/pack: constructed elsewhere (a function argument, a
  // returned pack, from_pointer_pack, an empty list, etc.). There are no
  // individual values to track; take the origin from the container type.
  extraOrigin = getVariadicContainerOrigin(callArgVal);
  assert(extraOrigin && "unknown variadic value source");
  return result;
}

//===----------------------------------------------------------------------===//
// CachedOriginFinder
//===----------------------------------------------------------------------===//

/// Unpack the specified value of OriginType into a set of referenced
/// origins. Returns true if any origins were found.
static bool handleOriginAttr(TypedAttr attr,
                             SmallVectorImpl<TypedAttr> &results) {
  bool foundAny = false;

  // Look through unions to find the values referenced.
  processOriginUnionElts(attr, [&](TypedAttr raw) {
    // FIXME(origins): This shouldn't happen; UncheckedCallEmission isn't
    // forming captures correctly for async functions with implicit origin
    // refs.
    //
    // ParamIndexRefAttr is only meaningful relative to an un-parameterized
    // generator scope.
    if (isa<ImplicitOriginRefAttr, ParamIndexRefAttr>(
            OriginMutCastAttr::strip(raw)))
      return;

    results.push_back(raw);
    foundAny = true;
  });
  return foundAny;
}

template <typename TypeOrAttr>
static bool scanForOrigins(TypeOrAttr typeOrAttr,
                           DenseSet<const void *> &typesAndAttrsWithoutOrigins,
                           DenseMap<const void *, bool> &visited,
                           SmallVectorImpl<TypedAttr> &results) {
  const void *rawPtr = typeOrAttr.getAsOpaquePointer();

  // Ignore types we have already scanned.
  if (typesAndAttrsWithoutOrigins.contains(rawPtr))
    return false;
  if (auto it = visited.find(rawPtr); it != visited.end())
    return it->second;

  // If this has origin type, process it.
  bool handled = false;
  bool hasOrigin = false;
  if constexpr (std::is_base_of_v<Attribute, TypeOrAttr>) {
    if (auto typedAttr = dyn_cast<TypedAttr>(typeOrAttr)) {
      if (isa<OriginType>(typedAttr.getType())) {
        hasOrigin |= handleOriginAttr(typedAttr, results);
        handled = true;
      }

      // If this is a parameter call or comptime load, only look at the result
      // type, not the completely arbitrary stuff that may be nested within it.
      // We don't care how the value was constructed, just whether the result
      // has origins.
      if (auto oper = dyn_cast<ParamOperatorAttr>(typedAttr)) {
        if (oper.getOpcode() == POC::ApplyResultSlot ||
            oper.getOpcode() == POC::Apply ||
            oper.getOpcode() == POC::LoadFromMem) {
          hasOrigin |= scanForOrigins(
              oper.getType(), typesAndAttrsWithoutOrigins, visited, results);
          handled = true;
        }
      }

      // Ignore sugar for the purpose of origin analysis, just look at the
      // canonical form. This is a compile time optimization.
      if (auto sugared = dyn_cast<SugarAttr>(typedAttr)) {
        hasOrigin |=
            scanForOrigins(sugared.getCanonical(), typesAndAttrsWithoutOrigins,
                           visited, results);
        handled = true;
      }
    }
  }

  // Values of function types should only consider any origins in the capture
  // set, not recursively nested types in the arguments or parameters defined by
  // the signature.
  if constexpr (std::is_base_of_v<Type, TypeOrAttr>) {
    if (auto fnType = dyn_cast<FnTypeGeneratorType>(typeOrAttr)) {
      hasOrigin |=
          scanForOrigins(fnType.getCaptureOrigins(),
                         typesAndAttrsWithoutOrigins, visited, results);
      handled = true;
    }
  }

  if (!handled) {
    // Recursively check for any nested types, e.g. the input/outputs of a
    // function type, types like !kgen.scalar<ty> etc.
    typeOrAttr.walkImmediateSubElements(
        [&](Attribute attr) {
          hasOrigin |= scanForOrigins(attr, typesAndAttrsWithoutOrigins,
                                      visited, results);
        },
        [&](Type type) {
          hasOrigin |= scanForOrigins(type, typesAndAttrsWithoutOrigins,
                                      visited, results);
        });
  }

  // If we can prove that this subtree doesn't contain origins, then remember
  // this so we don't revisit this type/attribute in the future.
  if (!hasOrigin)
    typesAndAttrsWithoutOrigins.insert(rawPtr);
  else
    // We don't need to visit the same attribute more than once to find origin
    // references. This is required to prevent splatting the parameter
    // expression tree.
    visited.try_emplace(rawPtr, hasOrigin);
  return hasOrigin;
}

/// This method finds all the origins buried in the specified type,
/// returning them as a list.  This typically will return ParamRefAttr's or
/// ImmutCast(ParamRefAttr)'s if a mutable origin is accessed immutably.
SmallVector<TypedAttr>
CachedOriginFinder::findOriginsIn(ArrayRef<Type> types,
                                  ArrayRef<TypedAttr> captures) {
  SmallVector<TypedAttr> results;

  // Scan each type, accumulating the results; the set avoid revisiting nodes
  // that we know cannot have origins.
  DenseMap<const void *, bool> visited;
  for (Type type : types)
    scanForOrigins(getCanonicalType(type), typesAndAttrsWithoutOrigins, visited,
                   results);
  for (TypedAttr capture : captures)
    scanForOrigins(getCanonicalAttr(capture), typesAndAttrsWithoutOrigins,
                   visited, results);
  return results;
}
