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
// Dependency-light per-target metadata, dispatched by triple via
// `TargetTraitsRegistry`. It lives in the low `Mojo/lib/Target` layer so any
// KGEN component can query a target's metadata without linking the codegen
// layer (`TargetBackend`). A supported target registers a full implementation
// from its own source file; a triple with no registered traits is an error at
// the use site (targets are dropped from a build by omitting their source).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_TARGETTRAITS_H
#define KGEN_TARGET_TARGETTRAITS_H

#include "Mojo/KGENDialect/KGENEnums.h"
#include "Support/ErrorOr.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class Triple;
class raw_ostream;
template <class C>
struct object_creator;
} // namespace llvm

namespace M::KGEN {

/// Per-target metadata dispatched by triple. Carries only cheap, broadly-useful
/// facts about a target (output-file extensions, accelerator arch tables); the
/// heavy codegen behavior stays on `TargetBackend` and the MLIR-lowering hooks
/// on `TargetLowering`.
class TargetTraits {
public:
  virtual ~TargetTraits() = default;

  /// Short name of the target, e.g. "host".
  virtual llvm::StringRef name() const = 0;
  /// Whether these traits describe `triple`.
  virtual bool matches(const llvm::Triple &triple) const = 0;

  /// Resolves `triple` to the concrete traits that describe it, or null if
  /// these do not. The default returns `this` when `matches`; a dispatcher that
  /// owns nested traits overrides this to return one of them.
  /// The registry prefers a resolution that differs from
  /// the traits it queried, so a nested traits object takes precedence over a
  /// direct match.
  virtual const TargetTraits *resolve(const llvm::Triple &triple) const {
    return matches(triple) ? this : nullptr;
  }

  /// Whether this target is a GPU.
  virtual bool isGPU() const { return false; }

  /// Default target CPU for `triple` when none is given, or empty to use the
  /// generic host/cross-compile fallback.
  virtual llvm::StringRef defaultCPU(const llvm::Triple &triple) const {
    return {};
  }

  /// Whether this target emits a standalone object file for offload kernels.
  /// Targets returning false have their emitted asm/IR file aliased under the
  /// object key instead (e.g. when the object toolchain is host-only).
  virtual bool emitsOffloadObjectFile() const { return true; }

  /// Required address space for a stack allocation on this target, or nullopt
  /// if the target does not constrain it.
  virtual std::optional<unsigned> requiredStackAllocationAddressSpace() const {
    return std::nullopt;
  }

  /// LLVM triple to use for this target's codegen, given `triple`. Defaults to
  /// `triple` unchanged; a target that compiles through a different LLVM triple
  /// normalizes it here.
  virtual std::string codegenTriple(llvm::StringRef triple) const {
    return triple.str();
  }

  /// Bitcode format version this target requires (an LLVM major version), or 0
  /// to use the caller's default. Overrides the caller-selected version.
  virtual unsigned forcedBitcodeVersion() const { return 0; }

  /// File extension for this target's assembly output (e.g. ".s").
  virtual llvm::StringRef getAsmExtension() const = 0;
  /// File extension for this target's LLVM IR output. Each target uses a
  /// distinct spelling so offload kernels from different targets do not
  /// collide in one output directory.
  virtual llvm::StringRef getLLVMExtension() const = 0;
  /// File extension for this target's object output (e.g. ".o").
  virtual llvm::StringRef getObjectExtension() const = 0;

  /// File extension for this target's LLVM bitcode output.
  virtual llvm::StringRef getBitcodeExtension() const = 0;

  /// One emission kind accepted by `--emit` for this target.
  struct EmissionKind {
    llvm::StringRef kind;
    /// What the compiler emits for this kind; link products (exe,
    /// shared-lib) compile to OBJECT, the linking is on the tool.
    EmitAs emitAs;
    llvm::StringRef description;
  };

  /// The emission kinds this target accepts, in display order; `-emit`
  /// validation rejects anything else. Dispatchers return an empty list and
  /// are omitted from the `--help` emission-kind table.
  virtual llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const = 0;

  /// The emission kinds every code-generating target shares. Targets return
  /// this directly or append their own kinds; identical descriptions merge
  /// into one row in the `--help` table.
  static llvm::ArrayRef<EmissionKind> commonEmissionKinds();

  /// Whether this target accepts `kind` as an emission kind. Non-virtual:
  /// derived from `supportedEmissionKinds` so acceptance cannot diverge
  /// from the declared table.
  bool supportsEmissionKind(llvm::StringRef kind) const;

  /// Errors if this target does not accept `kind` as an emission kind,
  /// naming the kinds it does accept. Non-virtual, as above.
  ErrorOrSuccess validateEmissionKind(llvm::StringRef kind) const;

  /// The EmitAs this target's `kind` compiles to; errors like
  /// `validateEmissionKind` if the target does not accept `kind`.
  ErrorOr<EmitAs> emitAsForKind(llvm::StringRef kind) const;

  /// One accelerator architecture accepted by `--target-accelerator`.
  struct AcceleratorArch {
    llvm::StringRef arch;
    llvm::StringRef description;
  };

  /// Title of this target's section in the `--print-supported-accelerators`
  /// table. Targets with no accelerator archs return an empty title and are
  /// omitted from the table.
  virtual llvm::StringRef acceleratorSectionTitle() const { return {}; }
  /// The accelerator archs this target accepts, in display order.
  virtual llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const {
    return {};
  }

protected:
  /// Check if the specific Target is a base target.
  /// This is needed to gate max only targets for now.
  virtual bool isBaseTarget() const = 0;

private:
  // The registry reads `isBastTarget()` to gate max only targets.
  friend class TargetTraitsRegistry;
};

/// Registry of `TargetTraits`, dispatched by triple. Mirrors
/// `TargetLoweringRegistry` and `TargetBackendRegistry`.
class TargetTraitsRegistry {
public:
  static TargetTraitsRegistry &get();

  /// Registers a traits object, taking ownership.
  void add(std::unique_ptr<TargetTraits> traits);

  /// Returns the traits describing `triple`.
  ErrorOr<const TargetTraits *> lookup(const llvm::Triple &triple) const;
  llvm::ArrayRef<std::unique_ptr<TargetTraits>> targets() const {
    return Targets;
  }

private:
  TargetTraitsRegistry() = default;
  friend struct llvm::object_creator<TargetTraitsRegistry>;

  std::vector<std::unique_ptr<TargetTraits>> Targets;
};

/// Print one kind-sorted `kind (target, ...) - description` row per distinct
/// (kind, description) pair, for the tools' `--emit` help output.
void printSupportedEmissionKinds(llvm::raw_ostream &os, unsigned indent = 2);

/// Whether any registered target accepts `kind` as an emission kind.
bool isKnownEmissionKind(llvm::StringRef kind);

/// The traits listing `arch` among their accelerator archs, or null.
const TargetTraits *traitsForAcceleratorArch(llvm::StringRef arch);

/// Errors if not `isBastTarget` and MAX is not installed.
ErrorOrSuccess requireMaxForAccelerator(bool isMaxOnly);

/// Same gate for a non-empty `--target-accelerator`.
/// Aborts rather than returns an error.
void requireMaxForAcceleratorRequest(llvm::StringRef targetAccelerator);

/// Registers `TraitsT` at static-init, e.g.:
///   static RegisterTargetTraits<HostTraits> X;
template <typename TraitsT>
struct RegisterTargetTraits {
  RegisterTargetTraits() {
    TargetTraitsRegistry::get().add(std::make_unique<TraitsT>());
  }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_TARGETTRAITS_H
