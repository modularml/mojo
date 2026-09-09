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

#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetTraits.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace M;
using namespace KGEN;
using namespace std::string_literals;

CompilationOptions::CompilationOptions(
    unsigned optimizationLevel, DebugInfoLevel debugLevel,
    std::optional<DebugAtLevel> debugAtLevel, Sanitizers sanitizers,
    std::string targetTriple, std::string targetCpu, std::string targetFeatures,
    std::string targetAccelerator, int elaborationErrorLimit,
    bool elaborationErrorIncludePrelude,
    ErrorVerboseLevel elaborationErrorVerbose, unsigned elaborationMaxDepth,
    DebugInfoLanguage debugInfoLanguage, std::string searchPaths,
    SmallVector<std::string> extraSearchPaths)
    : optimizationLevel(optimizationLevel), debugLevel(debugLevel),
      debugAtLevel(debugAtLevel), sanitizers(sanitizers),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)),
      targetAccelerator(std::move(targetAccelerator)),
      debugInfoLanguage(debugInfoLanguage), searchPaths(searchPaths),
      extraSearchPaths(extraSearchPaths),
      elaborationErrorLimit(elaborationErrorLimit),
      elaborationErrorIncludePrelude(elaborationErrorIncludePrelude),
      elaborationErrorVerbose(elaborationErrorVerbose),
      elaborationMaxDepth(elaborationMaxDepth) {

  // An explicit `--target-accelerator` requires MAX; fail before any target
  // lookup.
  requireMaxForAcceleratorRequest(this->targetAccelerator);

  if (this->targetCpu.empty())
    setDefaultCPU();
}

llvm::CodeGenOptLevel CompilationOptions::getCodeGenOptLevel() const {
  if (auto level = llvm::CodeGenOpt::getLevel(optimizationLevel))
    return *level;
  // Default to "Aggressive" optimizations.
  return llvm::CodeGenOptLevel::Aggressive;
}

DebugInfo::EmissionKind CompilationOptions::getDIEmissionKind() const {
  switch (debugLevel) {
  case kNoDebug:
    return DebugInfo::EmissionKind::None;
  case kSynthetic:
  case kLineTablesOnly:
    return DebugInfo::EmissionKind::LineTablesOnly;
  case kFullDebugInfo:
    return DebugInfo::EmissionKind::Full;
  }
  llvm_unreachable("unhandled debug level");
}

ErrorOr<EnvAttr>
CompilationOptions::parseDefinesWithDefaults(MLIRContext *ctx,
                                             ArrayRef<std::string> defines) {
  // Add defaults from compilation options.  Add them as strings before parsing
  // so that if a user defines them as well, they get an error for defining it a
  // second time.
  SmallVector<std::string> definesWithDefaults;
  switch (debugLevel) {
  case kFullDebugInfo:
    definesWithDefaults.push_back("__DEBUG_LEVEL=full");
    break;
  case kLineTablesOnly:
    definesWithDefaults.push_back("__DEBUG_LEVEL=line-tables");
    break;
  default:
    break;
  }
  definesWithDefaults.push_back("__OPTIMIZATION_LEVEL=" +
                                Twine(optimizationLevel).str());
  definesWithDefaults.push_back(
      "__SANITIZE_ADDRESS="s +
      (sanitizers.has(Sanitizers::kAddress) ? "1" : "0"));
  for (std::string define : defines)
    definesWithDefaults.push_back(define);
  return EnvAttr::parseDefines(ctx, definesWithDefaults);
}

StringRef CompilationOptions::getDebugLevelString() const {
  switch (debugLevel) {
  case kFullDebugInfo:
    return "full";
  case kLineTablesOnly:
    return "line-tables";
  default:
    return "";
  }
}

void CompilationOptions::print(raw_ostream &os) const {
  os << "CompilationOptions { optimizationLevel: " << optimizationLevel;
  if (debugLevel != kNoDebug) {
    os << ", debugLevel: "
       << (debugLevel == kLineTablesOnly ? "line-tables"
           : debugLevel == kSynthetic    ? "synthetic"
                                         : "full");
  }
  if (debugAtLevel) {
    os << ", debugAtLevel: ";
    switch (*debugAtLevel) {
    case kDebugAtLLVM:
      os << "llvm";
      break;
    case kDebugUnset:
      // do nothing
      break;
    }
  }
  if (sanitizers) {
    os << ", sanitizers:";
    sanitizers.print(os);
  }

  os << ", relocModel: " << stringifyRelocationModel(relocModel);

  if (!targetAccelerator.empty())
    os << ", targetAccelerator: " << targetAccelerator;

  os << ", debugInfoLang: " << debugInfoLanguage;

  if (numThreads != 0)
    os << ", numThreads: " << numThreads;

  os << " }";
}

void CompilationOptions::setDefaultCPU() {
  llvm::Triple triple(targetTriple);
  // A registered target supplies its own default CPU (e.g. the host target's
  // ARM baseline, or a plugin target); otherwise use the host/cross fallback.
  ErrorOr<const TargetTraits *> traitsOr =
      TargetTraitsRegistry::get().lookup(triple);
  const TargetTraits *traits = traitsOr.isError() ? nullptr : *traitsOr;
  llvm::StringRef targetDefault =
      traits ? traits->defaultCPU(triple) : llvm::StringRef();
  if (!targetDefault.empty()) {
    targetCpu = targetDefault.str();
  } else if (triple.getArch() !=
             llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch()) {
    // When cross-compiling, the host CPU is invalid for the target arch.
    // Clear it so LLVM selects the target's baseline CPU instead.
    targetCpu = "";
  } else {
    // Native target with no explicit CPU: use the host CPU.
    // TODO: reconsider this to maybe set a more conservative default.
    // Current behavior is that running `mojo build` on a host
    // generates a binary that cannot be run on lower-spec CPUs of the
    // same architecture.
    targetCpu = llvm::sys::getHostCPUName();
  }
}

bool M::KGEN::isGPUTriple(const llvm::Triple &triple) {
  // GPU-ness lives in the registered TargetTraits, so dropping a target's
  // sources drops it here too.
  ErrorOr<const TargetTraits *> traitsOr =
      TargetTraitsRegistry::get().lookup(triple);
  const TargetTraits *traits = traitsOr.isError() ? nullptr : *traitsOr;
  return traits && traits->isGPU();
}

bool M::KGEN::isMetalTriple(const llvm::Triple &triple) {
  // Metal GPU targets use ARM64 during compilation, then get converted to AIR
  // iOS/tvOS/watchOS don't have discrete GPUs suitable for compute kernels
  StringRef tripleStr = triple.str();

  return tripleStr.starts_with("air64-");
}

bool M::KGEN::overrideExported(const llvm::Triple &triple) {
  return isGPUTriple(triple);
}

bool M::KGEN::overrideExported(const CompilationOptions &options) {
  return overrideExported(llvm::Triple(options.targetTriple));
}
