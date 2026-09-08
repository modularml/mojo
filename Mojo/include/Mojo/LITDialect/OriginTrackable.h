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

#ifndef KGEN_LITDIALECT_ORIGIN_TRACKABLE_H
#define KGEN_LITDIALECT_ORIGIN_TRACKABLE_H

#include "Mojo/LITDialect/LITAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace M::KGEN {
namespace LIT {

class CachedOriginFinder {
public:
  /// This method finds all the origins buried in the specified type,
  /// returning them as a list, possibly eliding duplicates. This typically will
  /// return ParamRefAttr's or ImmutCast(ParamRefAttr)'s if a mutable origin
  /// is accessed immutably.
  SmallVector<TypedAttr> findOriginsIn(ArrayRef<Type> types,
                                       ArrayRef<TypedAttr> captures = {});

private:
  llvm::DenseSet<const void *> typesAndAttrsWithoutOrigins;
};

/// This class provide an abstraction for analyzing origin-trackable values,
/// e.g. variable definitions and owned arguments to functions.  This class can
/// also be used to query whether something is origin trackable or not, by
/// building a OriginTrackable and then querying it for null.
struct OriginTrackable {
  /// This constructor checks to see if the value is trackable, and if so
  /// identifies it.  If not, this returns a null value.
  OriginTrackable(Value value);

  /// This value feels true'y when it is initialized by something that can be
  /// origin tracked.
  operator bool() const { return !!name; }

  /// This is the user's declared name for the value declaration, or null if
  /// this isn't a tracked value.
  StringAttr name;

  /// This is true if the SSA value is a pointer to the logical storage instead
  /// of being the value itself.  This is always true for values of memory-only
  /// type.
  bool isIndirect = false;

  /// This is true if the value is uninitialized at function entry, false if it
  /// starts out initialized.
  bool startsUninit = false;

  /// This enum indicates the expected initialization state of a value upon
  /// return from a function. A function can return normally or return in an
  /// error state.
  enum ExitInitState {
    /// Value is never initialized upon function exit.
    EndsUninit,
    /// Value is always initialized upon function exit (e.g. as with a mut
    /// argument).
    EndsInit,
    /// Value is initialized upon a normal function exit (e.g. as with a
    /// byref_result argument).
    InitOnNormal,
    /// Value is initialized upon an error function exit (e.g. as with a
    /// byref_error argument).
    InitOnError
  };

  /// The expected initialization state of the value upon exit from a function.
  ExitInitState endInitState = ExitInitState::EndsUninit;

  /// True if this is a byref_result argument on an initializer: the self
  /// argument of an __init__ method.  These have magic behavior so
  /// they become fully initialized when all their fields are initialized.
  bool isFullObjectLiveOnEntry = false;

  //===----------------------------------------------------------------------===//
  // Helper methods
  //===----------------------------------------------------------------------===//

  /// This checks to see if the value is trackable or a field of a
  /// trackable.  If so it identifies the underlying object being referenced. If
  /// not, this returns a null value.
  static Value findUnderlyingValueFromField(Value value);

  /// Given the argument passed to a variadic argument, dig out the trackable
  /// values passed to the VariadicList/VariadicPack constructor.  Each of these
  /// may be used or consumed (by an owning variadic) so CheckLifetimes needs
  /// understand the impact on the individual arguments.
  ///
  /// This returns empty for forwarded containers, because there are no
  /// individual values to track.
  ///
  /// 'extraOrigin' is a hack because VariadicList doesn't track the origin of
  /// the vardecl it reads. This returns it so clients can know about it when
  /// needed.
  static SmallVector<Value>
  decodeIndividualVariadicArguments(Value value, TypedAttr &extraOrigin);
};

//===----------------------------------------------------------------------===//
// OperationValueEffects
//===----------------------------------------------------------------------===//

enum class ResultEffect {
  /// This is an ignorable result value, e.g. a value of trivial type.
  ignore,

  /// The result defines a new reg value, e.g. an owned register  result of a
  /// function call.
  regDefine,

  /// The result is a ref that starts uninitialized when defined, but is
  /// initialized by the end of the function.
  memDefineUninitToInit,

  /// The result is a ref that starts uninitialized when defined, and is also
  /// uninitialized by the end of the function.
  memDefineUninitToUninit,

  /// The result is a ref that starts initialized when defined, and is also
  /// initialized by the end of the function.
  memDefineInitToInit,

  /// The result is a ref that starts initialized when defined, but is
  /// uninitialized by the end of the function.
  memDefineInitToUninit,
};

enum class OperandEffect {
  /// This reads a register value and uses it, but does not consume it, e.g.
  /// a borrowed_reg argument.
  regUse,

  /// This takes ownership of an inreg value, e.g. owned_reg argument or
  /// RefStoreOp (which transfers ownership from the operand to the memory).
  regConsume,

  /// This is used by operations that load the value, things like RefLoadOp,
  /// LoadConsumeOp, OwnershipUseOp, MemcpyOp, and passing a borrowed operand.
  memLoad,

  /// This is store to the pointer that overwrites whatever is in it with a new
  /// owned value.  For example, RefStoreOp, MemcpyOp and ByRefResult call
  /// operands all do this.
  memStoreOwned,

  /// mut arg to a function call.  Value must be initialized before the
  /// operation, may be mutated, but then is still live afterward.
  memMut,

  /// This loads a value from the operand and takes ownership of the result, for
  /// example, owned operands (e.g. __del__) and LoadConsume.
  memConsume,

  /// This indicates that the full-object should be considered destroyed, but
  /// any fields within it are still valid.
  memMarkDestroyed,
};

/// This is the result value of `OperationEffects::analyze`, indicating
/// out-of-bound effects (aka special cases) and whether the op is unknown.
enum class OverallOpValueEffect {
  /// this indicates that the returned value effects cover everything.
  allHandled = 0,

  /// This is returned when the operation is unknown.
  unknownOp,

  /// This is a terminator op like return or unreachable.
  terminatorOp,

  /// This is HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, which all
  /// perform local control flow.
  localControlFlowOp,

  /// This is HLCF::IfOp, ParamIfOp, which are all if-like.
  ifLikeOp,

  /// This is HLCF::ElifOp specifically.
  elifOp,

  /// This is HLCF::LoopOp.
  loopOp,

  /// This is LIT::TryOp.
  tryOp,
};

/// Operand, result, and origin effects produced by `OperationEffects::analyze`.
struct OperationEffects {
  SmallVector<std::pair<Value, OperandEffect>> operands;
  SmallVector<ResultEffect> results;
  SmallVector<std::pair<TypedAttr, Value>> origins;
  /// Interior origins guaranteed live by this operation, e.g. a call to a
  /// callee whose declared return type contains an interior origin.
  SmallVector<InteriorOriginAttr> interiorOriginsDefined;

  OperationEffects(CachedOriginFinder &originFinder)
      : originFinder(originFinder) {}

  /// Computes the effects that `op` has on operands, result values, and other
  /// declared origins. Used by both phases of CheckLifetimes.
  OverallOpValueEffect analyze(Operation &op);

private:
  CachedOriginFinder &originFinder;

  void analyzeCallOp(Operation &op);
};

} // namespace LIT
} // namespace M::KGEN

#endif // KGEN_LITDIALECT_ORIGIN_TRACKABLE_H
