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
// Per-target policy for the MLIR-level KGEN/POP -> LLVM-dialect lowering. This
// is the lowering-stage counterpart to ObjectCompiler's `TargetBackend` (the
// codegen-stage abstraction): it lives in KGENToLLVM because the lowering
// passes here are below the codegen layer and cannot depend on `TargetBackend`.
// Both are dispatched by triple, so a target is described once per stage.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H
#define KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H

#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetTraits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class Triple;
template <class C>
struct object_creator;
} // namespace llvm

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class NamedAttrList;
class OpPassManager;
class Operation;
class RewritePatternSet;
class SymbolTable;
} // namespace mlir

namespace M::DebugInfo {
class DIType;
} // namespace M::DebugInfo

namespace M::KGEN {

/// Per-target policy for the KGEN/POP -> LLVM-dialect lowering, dispatched by
/// triple via `TargetLoweringRegistry`. A target contributes its own rewrite
/// patterns (the same way a compiler plugin already does); the default
/// contributes none (host-like).
///
/// As the inline per-target branches in the lowering passes are migrated, the
/// per-target logic moves into target-owned patterns registered through these
/// hooks, and the passes just call
/// `TargetLoweringRegistry::lookup(triple)->populate...`.
class TargetLowering {
public:
  virtual ~TargetLowering() = default;

  /// Cheap per-target metadata, or null on a dispatcher lowering that resolves
  /// to another lowering. Concrete targets override this; the default
  /// `name()`/`matches()` delegate to it.
  virtual const TargetTraits *traits() const { return nullptr; }

  /// Short name of the target lowering, used for diagnostics. Defaults to the
  /// traits' name; a dispatcher lowering overrides this directly.
  virtual llvm::StringRef name() const {
    return traits() ? traits()->name() : llvm::StringRef();
  }
  /// Whether this lowering handles `triple`. Defaults to the traits' match; a
  /// dispatcher lowering overrides this directly.
  virtual bool matches(const llvm::Triple &triple) const {
    return traits() && traits()->matches(triple);
  }

  /// Resolves `triple` to the concrete lowering that should handle it, or null
  /// if this one does not. The default returns `this` when `matches`; a
  /// lowering that owns nested lowerings (e.g. one discovered at runtime) can
  /// override this to return one of them. The registry prefers a resolution
  /// that differs from the lowering it queried, so a nested lowering takes
  /// precedence over a direct match.
  virtual const TargetLowering *resolve(const llvm::Triple &triple) const {
    return matches(triple) ? this : nullptr;
  }

  /// Contributes this target's patterns to the POP -> LLVM lowering
  /// (LowerPOPToLLVM): e.g. casts that lower to target-specific ISA intrinsics.
  /// The default contributes none. Mirrors the plugin's
  /// `populateLowerPOPToLLVMPatterns`, so a plugin is just another target
  /// lowering. Called after the generic patterns; target patterns use a higher
  /// benefit so they win for the ops they handle.
  virtual void
  populateLowerPOPToLLVMPatterns(mlir::RewritePatternSet &patterns,
                                 mlir::LLVMTypeConverter &converter,
                                 TargetInfoAttr target) const {}

  /// Contributes this target's patterns to the global (module-scoped) POP ->
  /// LLVM lowering (LowerGlobalPOPToLLVM), i.e. lowerings that create or query
  /// module-level symbols and therefore need a `SymbolTable`. The default
  /// contributes none. Mirrors the plugin's
  /// `populateLowerGlobalPOPToLLVMPatterns`, so a plugin is just another target
  /// lowering.
  virtual void populateLowerGlobalPOPToLLVMPatterns(
      mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &converter,
      mlir::SymbolTable &symtab, TargetInfoAttr target) const {}

  /// Contributes target-specific MLIR passes that run on the LLVM-dialect IR
  /// after the generic lowering pipeline and before translation to LLVM IR
  /// (e.g. late passes provided by a compiler plugin). The default adds none.
  virtual void addPostLowerToLLVMPasses(mlir::OpPassManager &pm) const {}

  /// Returns whether `op` is lowered by this target in the global POP->LLVM
  /// pass rather than the regular one (used to route the op via
  /// conversion-target legality). Default false.
  virtual bool isLoweredInGlobalPOPPass(mlir::Operation *op) const {
    return false;
  }

  /// Returns whether `op` is a target-specific operation that makes the
  /// enclosing function convergent (e.g. a GPU barrier). Used by the
  /// infer-function-attrs pass to propagate the `convergent` attribute.
  /// Default false.
  virtual bool isConvergentOp(mlir::Operation *op) const { return false; }

  /// Marks `func` (an exported `llvm.func` lowered for this target) with any
  /// target-specific attribute or calling convention that identifies it as a
  /// kernel entry point to the backend. The default does nothing, which is
  /// correct for targets with no kernel concept. Called from LowerKGENToLLVM
  /// when lowering an exported function.
  virtual void markExportedKernel(mlir::Operation *func) const {}

  /// Returns whether `func` is an exported kernel for this target, i.e. it
  /// carries the marking applied by `markExportedKernel`. The default returns
  /// false: targets with no kernel concept have no exported kernels.
  virtual bool isExportedKernel(mlir::Operation *func) const { return false; }

  /// Returns the LLVM argument-attribute name this target uses to pass a
  /// borrowed pointer argument to a kernel by value. Only queried for targets
  /// whose `isExportedKernel` returns true, so the default is empty.
  virtual llvm::StringRef getKernelByValArgAttrName() const { return {}; }

  /// Returns the memory-form type this target requires for a kernel argument of
  /// `type` that would otherwise be passed in registers, or a null type if it
  /// imposes no such requirement. The default imposes nothing.
  virtual mlir::Type lowerKernelArgToMemory(mlir::Type type) const {
    return {};
  }

  /// Returns the indirected (wrapped) type this target requires for a kernel
  /// argument of `type` with calling `convention`, or a null type if none. The
  /// default imposes nothing.
  virtual mlir::Type
  getKernelArgIndirectionType(mlir::Type type, ArgConvention convention) const {
    return {};
  }

  /// Returns whether the KGEN verifier should run this target's `verifyOp`
  /// checks. The default is false.
  virtual bool needsVerification() const { return false; }

  /// Verifies that `op` is legal on this target, emitting a diagnostic on `op`
  /// and returning failure if not. Only called when `needsVerification` is
  /// true. The default accepts everything.
  virtual mlir::LogicalResult verifyOp(mlir::Operation *op) const {
    return mlir::success();
  }

  /// Translates a kernel-argument annotation `attr` (carried via the
  /// `@__llvm_metadata` decorator) into LLVM argument attributes on `argAttrs`.
  /// A target overrides this to apply the annotations it recognizes.
  ///
  /// The default drops the annotation: an annotation a target does not
  /// recognize is silently ignored, keeping the same mojo code portable across
  /// backends.
  /// TODO: replace the drop-by-default behavior with a more precise per-target
  /// policy that distinguishes a foreign-but-known annotation from a genuinely
  /// unsupported one.
  virtual void mapKernelArgMetadata(mlir::Operation *func,
                                    mlir::NamedAttribute attr,
                                    mlir::NamedAttrList &argAttrs) const {}

  /// Applies any target-specific finalization to `func` after the KGEN -> LLVM
  /// dialect conversion completes. The default does nothing.
  virtual void finalizeConvertedFunction(mlir::Operation *func) const {}

  /// Translates a target-dialect function-level metadata attribute `attr`
  /// (carried via the `@__llvm_metadata` decorator) into LLVM function
  /// attributes (`funcAttrs`) or passthrough entries (`passthrough`). Sets
  /// `handled` to true when the attribute was consumed; leaves it false to let
  /// the caller fall back to generic passthrough handling. Returns failure
  /// (with an error emitted on `func`) if the attribute is recognized but
  /// malformed. The default handles nothing.
  virtual mlir::LogicalResult
  mapFuncMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                  mlir::NamedAttrList &funcAttrs,
                  llvm::SmallVectorImpl<mlir::Attribute> &passthrough,
                  bool &handled) const {
    handled = false;
    return mlir::success();
  }

  /// Contributes the dtype conversions this target supports natively (beyond
  /// the common LLVM fpext/fptrunc set), by calling `addConversion(from, to)`
  /// for each. Used by the POP legalizer to decide which casts need software
  /// widening. The default contributes none.
  virtual void populateLegalConversions(
      TargetInfoAttr target,
      llvm::function_ref<void(KGENDType from, KGENDType to)> addConversion)
      const {}

  /// Returns whether `dtype` needs legalization on this target beyond the
  /// generic "float width <= 8 bits" rule the legalizer already applies (e.g. a
  /// target that lacks native arithmetic for some 16-bit float). Default false.
  virtual bool typeNeedsExtraLegalization(KGENDType dtype,
                                          TargetInfoAttr target) const {
    return false;
  }

  /// Returns whether a `pop.cast` between `fromDType` and `toDType` is handled
  /// by a later target-specific pass (so the POP legalizer should skip it
  /// here). Default false.
  virtual bool castHandledByLaterPass(KGENDType fromDType, KGENDType toDType,
                                      TargetInfoAttr target) const {
    return false;
  }

  /// Returns a target-specific spelling of `dtype`'s name for debug info, or
  /// nullopt to use the generic encoding (e.g. a target's debugger may expect
  /// language-specific type names). The default has no preference.
  virtual std::optional<std::string> getDTypeDebugName(KGENDType dtype) const {
    return std::nullopt;
  }

  /// Returns a target-specific debug type for `dtype`, or a null type to use
  /// the generic encoding (e.g. a target's debugger may expect certain dtypes
  /// as specially-named struct types). The default returns a null type.
  virtual DebugInfo::DIType buildDebugTypeForDType(mlir::MLIRContext *ctx,
                                                   KGENDType dtype) const;

protected:
  /// Check if the specific Target is a base target.
  /// This is needed to gate max only targets for now.
  virtual bool isBaseTarget() const = 0;

private:
  // The registry reads `isBaseTarget()` to gate accelerator targets on MAX.
  friend class TargetLoweringRegistry;
};

/// Registry of `TargetLowering`s, dispatched by triple. Lowering-stage
/// counterpart to `TargetBackendRegistry`.
class TargetLoweringRegistry {
public:
  static TargetLoweringRegistry &get();

  /// Registers a lowering, taking ownership.
  void add(std::unique_ptr<TargetLowering> lowering);
  /// Returns the lowering for `triple` (never null on success). Errors when
  /// `triple` is an accelerator without MAX, or none is registered.
  ErrorOr<const TargetLowering *> lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetLowering>> targets() const {
    return Targets;
  }

private:
  TargetLoweringRegistry() = default;
  friend struct llvm::object_creator<TargetLoweringRegistry>;

  std::vector<std::unique_ptr<TargetLowering>> Targets;
};

/// Registers `LoweringT` at static-init, e.g.:
///   static RegisterTargetLowering<MyTargetLowering> X;
template <typename LoweringT>
struct RegisterTargetLowering {
  RegisterTargetLowering() {
    TargetLoweringRegistry::get().add(std::make_unique<LoweringT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_KGENTOLLVM_TARGET_TARGETLOWERING_H
