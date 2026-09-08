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

#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Support/Configuration.h"
#include "Target/TargetTraits.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"

namespace M::KGEN {

void addAddressSanitizerPass(llvm::ModulePassManager &mpm,
                             const CompilationOptions &options) {
  if (!options.sanitizers.has(M::Sanitizers::kAddress))
    return;
  llvm::AddressSanitizerOptions opts;
  bool moduleUseAfterScope = false;
  bool useOdrIndicator = false;
  mpm.addPass(
      llvm::AddressSanitizerPass(opts, moduleUseAfterScope, useOdrIndicator));
}

void addThreadSanitizerPass(llvm::ModulePassManager &mpm,
                            const CompilationOptions &options) {
  if (!options.sanitizers.has(M::Sanitizers::kThread))
    return;
  mpm.addPass(llvm::ModuleThreadSanitizerPass());
  mpm.addPass(
      llvm::createModuleToFunctionPassAdaptor(llvm::ThreadSanitizerPass()));
}

void TargetBackend::addSanitizers(llvm::ModulePassManager &mpm,
                                  const CompilationOptions &options) const {
  addAddressSanitizerPass(mpm, options);
  addThreadSanitizerPass(mpm, options);
}

void TargetBackend::emitBitcode(llvm::Module &module,
                                llvm::raw_pwrite_stream &os) const {
  llvm::WriteBitcodeToFile(module, os, /*ShouldPreserveUseListOrder=*/true);
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
defaultCreateTargetMachine(const CompilationOptions &options, bool isJIT) {
  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(options.targetTriple), errorMessage);
  if (!target) {
    return Error("no target exists for '" + options.targetTriple +
                 "': " + errorMessage);
  }

  // The ABI name can drive the target data layout, so it must be set at
  // machine creation time. `targetABI` takes precedence over the user-facing
  // `targetAbi` since it encodes a data-layout invariant, not just a
  // calling-convention preference.
  llvm::TargetOptions targetOptions;
  targetOptions.MCOptions.ABIName =
      !options.targetABI.empty() ? options.targetABI : options.targetAbi;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(options.targetTriple), options.targetCpu,
      options.targetFeatures, targetOptions, options.relocModel,
      /*CM=*/options.mcmodel,
      /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  if (options.largeDataThreshold)
    machine->setLargeDataThreshold(options.largeDataThreshold.value());

  return machine;
}

bool TargetBackend::isSharedMemoryGlobal(
    const llvm::GlobalVariable &global) const {
  std::optional<unsigned> addressSpace = sharedMemoryAddressSpace();
  if (!addressSpace || global.getAddressSpace() != *addressSpace)
    return false;

  // A "._gpu_shared_mem" name handle confirms the global came from a
  // stack_allocation in shared memory.
  // Externally linked globals in the shared
  // address space are also treated as shared memory (not split anchors).
  return global.getName().contains("._gpu_shared_mem") ||
         global.hasExternalLinkage();
}

static llvm::ManagedStatic<TargetBackendRegistry> theBackendRegistry;

TargetBackendRegistry &TargetBackendRegistry::get() {
  return *theBackendRegistry;
}

void TargetBackendRegistry::add(std::unique_ptr<TargetBackend> backend) {
  Backends.push_back(std::move(backend));
}

ErrorOr<const TargetBackend *>
TargetBackendRegistry::lookup(const llvm::Triple &triple) const {
  // A backend that resolves `triple` to one it owns takes precedence over a
  // direct self-match.
  const TargetBackend *result = [&]() -> const TargetBackend * {
    const TargetBackend *directMatch = nullptr;
    for (const std::unique_ptr<TargetBackend> &backend : Backends) {
      const TargetBackend *resolved = backend->resolve(triple);
      if (!resolved)
        continue;
      if (resolved != backend.get())
        return resolved;
      if (!directMatch)
        directMatch = resolved;
    }
    return directMatch;
  }();
  if (!result) {
    return Error("target '" + triple.str() +
                 "' is not supported by this build");
  }
  if (ErrorOrSuccess e = requireMaxForAccelerator(!result->isBaseTarget()))
    return Error(e.getError());
  return result;
}

} // namespace M::KGEN
