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
// Per-target policy for LLVM-level codegen in ObjectCompiler: module splitting,
// runtime-library linking, object/asm emission, and packaging. Host and each
// GPU vendor are peer `TargetBackend`s, dispatched by triple via
// `TargetBackendRegistry`.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_TARGETBACKEND_H
#define KGEN_COMPILER_TARGET_TARGETBACKEND_H

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Target/TargetTraits.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"

#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class Function;
class GlobalVariable;
class Module;
class PassBuilder;
class TargetMachine;
class Triple;
class raw_pwrite_stream;
template <class C>
struct object_creator;
} // namespace llvm

namespace mlir {
class Operation;
} // namespace mlir

namespace M::KGEN {

class MCLinker;

/// Creates an LLVM TargetMachine from already-effective `options`.
/// Runs entirely in the host's LLVM instance (it uses the target registry,
/// which is only initialized here). Callers that need target-specific tweaks
/// should apply `TargetBackend::adjustOptionsForTargetMachine` first.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
defaultCreateTargetMachine(const CompilationOptions &options, bool isJIT);

/// `defaultCreateTargetMachine`'s signature. Passed to a backend loaded from a
/// separate shared library so `TargetMachine` creation runs against the host's
/// LLVM target registry; a separately-loaded library has its own registry,
/// which nothing initializes.
using CreateTargetMachineFn = ErrorOr<std::unique_ptr<llvm::TargetMachine>> (*)(
    const CompilationOptions &options, bool isJIT);

/// Adds the AddressSanitizer / ThreadSanitizer passes when requested in
/// `options`. Shared building blocks for `TargetBackend::addSanitizers` and the
/// no-backend fallback.
void addAddressSanitizerPass(llvm::ModulePassManager &mpm,
                             const CompilationOptions &options);
void addThreadSanitizerPass(llvm::ModulePassManager &mpm,
                            const CompilationOptions &options);

/// How the module is divided into independently-codegen'd units.
enum class SplitStrategy { None, PerExported, PerFunction };

/// Per-compilation state passed to the emit/package hooks.
struct EmitContext {
  /// Runs llc into `out`, emitting object code when `objectFile`, else asm.
  using RunLlc = llvm::function_ref<ErrorOrSuccess(
      llvm::Module &module, WriteableBuffer &out, bool createObjectFile)>;
  /// Links an emitted object into the final shared object/binary (plugin or
  /// lld).
  using LinkObject = llvm::function_ref<ErrorOr<BufferRef>(
      BufferRef object, llvm::StringRef moduleName)>;

  const CompilationOptions &options;
  llvm::TargetMachine &tm;
  mlir::Location loc;
  size_t moduleIdx = 0;
  MCLinker *linker = nullptr;
  /// Path to the system linker (lld), for backends that produce the final
  /// shared object themselves (e.g. a plugin-provided backend).
  llvm::StringRef linkerPath;
  RunLlc runLlc;
  LinkObject linkObject;
};

/// Immutable description of an LLVM-level compilation target.
class TargetBackend {
public:
  virtual ~TargetBackend() = default;

  /// Cheap per-target metadata (name, triple match, output extensions), or null
  /// on a dispatcher backend that resolves to another backend. Concrete targets
  /// override this; the default `name()`/`matches()` delegate to it.
  virtual const TargetTraits *traits() const { return nullptr; }

  /// Short name of the backend, e.g. "host", "nvptx". Defaults to the traits'
  /// name; a dispatcher backend overrides this directly.
  virtual llvm::StringRef name() const {
    return traits() ? traits()->name() : llvm::StringRef();
  }
  /// Whether this backend handles `triple`. Defaults to the traits' match; a
  /// dispatcher backend overrides this directly.
  virtual bool matches(const llvm::Triple &triple) const {
    return traits() && traits()->matches(triple);
  }

  /// Resolves `triple` to the concrete backend that should handle it, or null
  /// if this backend does not. The default returns `this` when `matches`; a
  /// backend that owns nested backends (e.g. one discovered at runtime) can
  /// override this to return one of them. The registry prefers a resolution
  /// that differs from the backend it queried, so a nested backend takes
  /// precedence over a direct match.
  virtual const TargetBackend *resolve(const llvm::Triple &triple) const {
    return matches(triple) ? this : nullptr;
  }

  virtual SplitStrategy
  splitStrategy(const CompilationOptions &options) const = 0;
  /// Inter-procedural codegen disables parallel/per-function llc.
  virtual bool isCodegenInterprocedural() const { return false; }

  /// Whether this backend produces offload (device) code; a missing kernel id
  /// is a hard error for offload targets.
  virtual bool isOffload() const { return false; }

  /// Whether `global` lives in this target's shared/threadgroup memory. Such
  /// globals are not split anchors: they are shared among threads within a
  /// block but never across kernels. The default uses
  /// `sharedMemoryAddressSpace()`; targets without shared memory (e.g. host)
  /// never match.
  virtual bool isSharedMemoryGlobal(const llvm::GlobalVariable &global) const;

protected:
  /// The target's shared/threadgroup-memory address space, or nullopt for
  /// targets without one (e.g. host). Backing datum for
  /// `isSharedMemoryGlobal`.
  virtual std::optional<unsigned> sharedMemoryAddressSpace() const {
    return std::nullopt;
  }

  /// Check if the specific Target is a base target.
  /// This is needed to gate max only targets for now.
  virtual bool isBaseTarget() const = 0;

public:
  /// Whether the linked module's functions must be restored to their original
  /// order after MCLinker reorders them (needed when codegen emits function
  /// declarations whose order matters, e.g. NVPTX).
  virtual bool requiresOriginalFunctionOrder() const { return false; }

  /// Whether the backend requires the module to be emitted as a single split
  /// (e.g. NVPTX, whose AnnotationCache must be shared across the llvm-opt and
  /// llc pipelines, so MCLinker cannot operate on multiple splits).
  virtual bool requiresSingleModuleSplit() const { return false; }

  /// Reorders `module`'s functions back to `originalOrder` after MCLinker
  /// linking, for backends that emit function declarations before uses
  /// (e.g. NVPTX). The default leaves the order untouched.
  virtual void reorderLinkedModuleFunctions(
      llvm::Module &module,
      const llvm::StringMap<unsigned> &originalOrder) const {}

  /// Adjusts the SimplifyCFG options for this target. The default leaves them
  /// unchanged; GPUs raise the bonus instruction threshold because branches are
  /// far costlier than on CPU.
  virtual llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const {
    return options;
  }

  /// Adds the sanitizer passes this target supports. The default applies the
  /// host policy (AddressSanitizer and ThreadSanitizer as requested in
  /// `options`); backends override to restrict (e.g. GPUs drop ThreadSanitizer,
  /// NVPTX drops all sanitizers).
  virtual void addSanitizers(llvm::ModulePassManager &mpm,
                             const CompilationOptions &options) const;

  /// Adjusts the MLIR module before lowering to LLVM (e.g. target-specific
  /// debug-info fixups). Runs in `ObjectCompiler::lowerAllFuncsToLLVM`.
  virtual void
  prepareModuleForLowering(mlir::Operation *module,
                           const CompilationOptions &options) const {}

  /// Adjusts `options` before `defaultCreateTargetMachine` builds the machine;
  /// the identity by default.
  virtual CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const {
    return options;
  }
  /// Fixes up `module` after the TargetMachine is created.
  virtual void finalizeModuleForTarget(llvm::Module &module,
                                       llvm::TargetMachine &tm,
                                       llvm::StringRef originalTriple) const {}

  /// Writes `module` as bitcode to `os`. The base implementation writes
  /// standard LLVM bitcode; backends may override (e.g. Metal emits AIR
  /// bitcode).
  virtual void emitBitcode(llvm::Module &module,
                           llvm::raw_pwrite_stream &os) const;

  /// Builds the backend's complete LLVM pass pipeline into `mpm`, returning
  /// true if it fully owns the pipeline (so the generic optimization pipeline
  /// is skipped). The default returns false; backends that need a fully custom
  /// pipeline (e.g. Metal's AIR legalization) override it.
  virtual bool buildLLVMPipeline(llvm::ModulePassManager &mpm,
                                 llvm::PassBuilder &passBuilder,
                                 const CompilationOptions &options) const {
    return false;
  }

  /// Adds backend-specific passes at the start of the standard optimization
  /// pipeline (for backends that augment rather than replace it, e.g. NVPTX).
  virtual void addPipelineStartPasses(llvm::ModulePassManager &mpm,
                                      const CompilationOptions &options) const {
  }

  /// Registers the backend's named passes with `passBuilder` for `-passes=`
  /// (used by kgen-llvm-opt). The default registers nothing.
  virtual void registerPipelinePasses(llvm::PassBuilder &passBuilder) const {}

  /// Links target-specific device/runtime bitcode into `module`.
  virtual ErrorOrSuccess
  linkRuntimeLibraries(mlir::Location loc, llvm::Module &module,
                       const CompilationOptions &options) const {
    return {};
  }

  /// Whether custom (non-runtime) bitcode libraries should be linked
  /// conservatively — pulling in only the symbols referenced by the module's
  /// unresolved extern functions (`LinkOnlyNeeded`), and skipping the link
  /// entirely when there are no such references. The default is true. Backends
  /// whose bitcode library must be linked in full regardless of which symbols
  /// are referenced return false.
  virtual bool onlyLinkExternFunctionsInBitcodeLibs(
      const CompilationOptions &options) const {
    return true;
  }

  /// Attaches target-specific attributes to a kernel entry point.
  virtual void attachCodegenAttributes(llvm::Function *kernelEntry) const {}

  /// Appends backend-specific arguments to the link step.
  virtual void appendLinkArgs(llvm::SmallVectorImpl<llvm::StringRef> &args,
                              const CompilationOptions &options) const {}

  virtual ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                          EmitContext &ctx) const = 0;
  virtual ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                        EmitContext &ctx) const = 0;
  /// Combines per-unit objects into the final artifact.
  virtual ErrorOr<BufferRef>
  createArchive(llvm::MutableArrayRef<BufferRef> objects,
                llvm::StringRef moduleName, EmitContext &ctx) const = 0;

  friend class TargetBackendRegistry;
};

/// Registry of `TargetBackend`s, dispatched by triple.
class TargetBackendRegistry {
public:
  static TargetBackendRegistry &get();

  /// Registers a backend, taking ownership.
  void add(std::unique_ptr<TargetBackend> backend);

  /// Returns the backend for `triple` (never null on success).
  ErrorOr<const TargetBackend *> lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetBackend>> backends() const {
    return Backends;
  }

private:
  TargetBackendRegistry() = default;
  friend struct llvm::object_creator<TargetBackendRegistry>;

  std::vector<std::unique_ptr<TargetBackend>> Backends;
};

/// Registers `BackendT` at static-init, e.g.:
///   static RegisterTargetBackend<XYZBackend> X;
template <typename BackendT>
struct RegisterTargetBackend {
  RegisterTargetBackend() {
    TargetBackendRegistry::get().add(std::make_unique<BackendT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_TARGETBACKEND_H
