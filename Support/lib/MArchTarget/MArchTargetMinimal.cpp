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

#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/X86TargetParser.h"
#include <cassert>
#include <clang/Basic/LLVM.h>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace M;

/// Returns the feature set according to clang for the given options, which
/// should include the Triple and CPU.
ErrorOr<std::vector<std::string>>
M::getFeaturesFromClang(std::shared_ptr<clang::TargetOptions> opts,
                        StringRef cpu) {
  // Intercept diagnostics from Clang and then bundle them up in an `Error` if
  // something bad happens.
  struct DiagInterceptor : public clang::DiagnosticConsumer {
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &info) override {
      if (level >= clang::DiagnosticsEngine::Level::Error) {
        // Keep the last message.
        msg.clear();
        info.FormatDiagnostic(msg);
      }
    }

    SmallString<64> msg;
  };

  // Instantiate the Clang diagnostic engine. Pass in our interceptor.
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> ids(
      new clang::DiagnosticIDs());
  clang::DiagnosticOptions diagOpts;
  DiagInterceptor interceptor;
  clang::DiagnosticsEngine diags(std::move(ids), diagOpts, &interceptor,
                                 /*ShouldOwnClient=*/false);

  // Ask Clang to create the target info for the architecture and CPU. This will
  // populate `opts` with the full target triple and feature set.
  auto targetInfo = std::unique_ptr<clang::TargetInfo>(
      clang::TargetInfo::CreateTargetInfo(diags, *opts.get()));
  if (!targetInfo)
    return Error("failed to create target info: " + interceptor.msg);

  std::vector<std::string> features;

  // AARCH64 CPU features are not parsed by `CreateTargetInfo`. We have to query
  // them and add them here manually.
  if (std::optional<llvm::AArch64::CpuInfo> cpuInfo =
          llvm::AArch64::parseCpu(cpu)) {
    std::vector<std::string> updatedFeaturesVec;
    auto exts = cpuInfo->DefaultExtensions;
    std::vector<StringRef> cpuFeats;
    llvm::AArch64::getExtensionFeatures(exts, cpuFeats);
    for (StringRef f : cpuFeats) {
      assert((f[0] == '+' || f[0] == '-') && "Expected +/- in target feature!");
      updatedFeaturesVec.push_back(f.str());
    }
    llvm::StringMap<bool> featureMap;
    targetInfo->initFeatureMap(featureMap, diags, cpu, updatedFeaturesVec);
    for (const auto &f : featureMap)
      opts->Features.push_back((f.getValue() ? "+" : "-") + f.getKey().str());
  }

  // RISC-V CPU models are not expanded by `CreateTargetInfo` either: clang
  // derives its extensions from `-march`, so a `-mcpu`-only invocation reports
  // just the base ISA. Resolve the CPU's default march instead, which also
  // pulls in the extensions it implies (`M` implies `Zmmul`, `D` implies `F`).
  // TODO(#6918): `-march` should win over `-mcpu` as in the clang driver. The
  // CPU defaults are unioned in twice -- here, and again by
  // `getTargetInfoFor`'s delta merge -- and un-unioning them means teaching
  // that shared merge to treat an explicit ISA string as complete. Until then,
  // `--march=rv32i --mtune=sifive-e31` yields exactly the requested ISA.
  if (llvm::Triple(opts->Triple).isRISCV()) {
    SmallVector<std::string> cpuFeats;
    llvm::RISCV::getFeaturesForCPU(cpu, cpuFeats, /*NeedPlus=*/true);
    llvm::append_range(opts->Features, cpuFeats);
  }

  // Concat the features together, only keeping included '+' features.
  for (StringRef feature : opts->Features) {
    if (feature.front() == '+')
      features.emplace_back(feature.drop_front());
  }

  llvm::sort(features);
  // The per-architecture blocks above append what clang already resolved, so a
  // CPU's base ISA can appear twice.
  features.erase(llvm::unique(features), features.end());

  return features;
}

/// Returns feature set for host, falling back to clang using triple and cpu
/// options if the native LLVM helper fails.
ErrorOr<std::vector<std::string>> M::getFeatures(StringRef triple,
                                                 StringRef cpu) {
  auto opts = std::make_shared<clang::TargetOptions>();
  opts->Triple = triple;
  opts->CPU = cpu;
  return getFeaturesFromClang(opts, cpu);
}

ErrorOr<M::ResolvedCpu> M::resolveCpu(StringRef triple, StringRef cpu) {
  ErrorOr<std::vector<std::string>> featuresOr = getFeatures(triple, cpu);
  if (!featuresOr.isError())
    return ResolvedCpu{cpu.str(), featuresOr.takeValue()};
  if (cpu.empty())
    return featuresOr.takeError();

  // LLVM answers "generic" for a CPU it cannot identify, and that name is not
  // universally valid: Clang checks it against a per-architecture TargetParser
  // CPU table, and x86_64's omits "generic" while AArch64's carries it. So an
  // unidentified host CPU takes target creation down on x86_64, and would on
  // any architecture whose table omits the name. Asking for no CPU name gives
  // the triple's baseline, which is what "unidentified" means.
  featuresOr = getFeatures(triple, "");
  if (featuresOr.isError())
    return featuresOr.takeError();
  return ResolvedCpu{"", featuresOr.takeValue()};
}

ErrorOr<TargetInfo> M::getHostTargetInfo() {
  std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  ErrorOr<ResolvedCpu> hostCpuOr =
      resolveCpu(hostTriple, llvm::sys::getHostCPUName());
  if (hostCpuOr)
    return hostCpuOr.takeError();
  ResolvedCpu hostCpu = hostCpuOr.takeValue();
  std::vector<std::string> features = std::move(hostCpu.features);

  // A Docker container or hypervisor may only enable a subset of CPU features
  // (e.g. no AVX-512). Use LLVM's runtime
  // CPUID detection to identify absent features and record them in
  // disabledFeatures so LLVM doesn't re-enable them from CPU model defaults.
  llvm::StringMap<bool> runtimeFeatures = llvm::sys::getHostCPUFeatures();
  std::vector<std::string> disabledFeatures{};
  if (!runtimeFeatures.empty()) {
    auto featureIsAbsent = [&](StringRef f) {
      auto it = runtimeFeatures.find(f);
      return it != runtimeFeatures.end() && !it->second;
    };
    for (StringRef f : features)
      if (featureIsAbsent(f))
        disabledFeatures.push_back(f.str());
    llvm::erase_if(features, featureIsAbsent);
  }

  return TargetInfo(llvm::Triple(hostTriple), hostCpu.name, std::move(features),
                    std::move(disabledFeatures));
}

std::string M::getHostCPUFeatures() {
  auto targetInfoOr = getHostTargetInfo();
  if (targetInfoOr)
    return "";
  return encodeFeatures(*targetInfoOr);
}

ErrorOr<TargetInfo> M::getMArchTargetInfo(StringRef targetTriple,
                                          StringRef march, StringRef mcpu,
                                          StringRef mtune) {
  using namespace llvm;

  // Handle -march=native.
  if (march == "native")
    return getHostTargetInfo();

  // `-march` has different meaning depending on the architecture. Determine the
  // LLVM target triple and CPU from it.
  Triple triple = llvm::Triple(targetTriple);
  auto opts = std::make_shared<clang::TargetOptions>();

  auto processExts = [&opts](StringRef &m) {
    StringRef exts;
    std::tie(m, exts) = m.split("+");
    while (!exts.empty()) {
      StringRef ext;
      std::tie(ext, exts) = exts.split("+");
      if (ext.starts_with("no"))
        opts->FeatureMap[ext.drop_front(2)] = false;
      else
        opts->FeatureMap[ext] = true;
    }
  };
  processExts(march);
  processExts(mcpu);

  if (!mtune.empty())
    opts->TuneCPU = mtune;

  auto tryParseX86 = [&](StringRef cpuName) {
    // Check for a 64-bit one first.
    // x86_64 is a well-known name for the architecture, so it is not subject to
    // our coding standard for variable names.
    // NOLINTBEGIN
    if (X86::CPUKind x86_64Cpu = X86::parseArchX86(cpuName, /*Only64Bit=*/true);
        x86_64Cpu != X86::CK_None) {
      // NOLINTEND
      triple.setArch(Triple::x86_64);
      opts->CPU = cpuName;
      return true;
    }

    // Otherwise, see if it is a 32-bit one.
    if (X86::CPUKind x86Cpu = X86::parseArchX86(cpuName, /*Only64Bit=*/false);
        x86Cpu != X86::CK_None) {
      triple.setArch(Triple::x86_64);
      opts->CPU = cpuName;
      return true;
    }

    return false;
  };

  // Try to parse an X86 architecture from either -march or -mcpu.
  if (tryParseX86(march) || tryParseX86(mcpu)) {
    if (mcpu == "generic")
      opts->CPU = "";

    // Check for an AArch64 CPU.
  } else if (std::optional<AArch64::CpuInfo> aarch64Cpu =
                 AArch64::parseCpu(mcpu)) {
    triple.setArch(Triple::aarch64, triple.getSubArch());
    opts->CPU = mcpu;

    // Check for an ARM CPU.
  } else if (ARM::ArchKind armArch = ARM::parseCPUArch(mcpu);
             armArch != ARM::ArchKind::INVALID) {
    triple.setArchName(ARM::getArchName(armArch));
    opts->CPU = mcpu;

    // Check for an AArch64 arch.
  } else if (AArch64::parseArch(march)) {
    triple.setArchName(march);
    opts->CPU = "generic";

    // Check for an ARM arch.
  } else if (ARM::ArchKind armArch = ARM::parseArch(march);
             armArch != ARM::ArchKind::INVALID) {
    triple.setArchName(march);
    // If -mcpu was not specified, use a default CPU for the architecture.
    if (mcpu.empty())
      opts->CPU = ARM::getDefaultCPU(triple.getArchName());
    else
      opts->CPU = mcpu;

    // The triple already names the architecture and a RISC-V CPU name does not
    // imply it, so unlike x86/ARM the triple is left alone.
    // Extensions come from the `-march` ISA string when given, otherwise from
    // the CPU's default march (resolved in `getFeaturesFromClang`).
  } else if (triple.isRISCV()) {
    opts->CPU = mcpu;
    if (!march.empty()) {
      // Experimental extensions are accepted without the driver's
      // `-menable-experimental-extensions` opt-in: the audience for `-march` is
      // firmware pinned to one core, and the ISA string still has to name the
      // explicit version of anything experimental.
      llvm::Expected<std::unique_ptr<RISCVISAInfo>> isaOr =
          RISCVISAInfo::parseArchString(march,
                                        /*EnableExperimentalExtension=*/true);
      if (llvm::Error err = isaOr.takeError()) {
        return Error("invalid RISC-V -march '" + march.str() +
                     "': " + llvm::toString(std::move(err)));
      }
      // `CreateTargetInfo` resolves `FeaturesAsWritten` and overwrites
      // `Features`, so the ISA string has to go in as written.
      llvm::append_range(opts->FeaturesAsWritten, (*isaOr)->toFeatures());
    }

  } else {
    triple.setArchName(march);
    opts->CPU = "generic";
  }

  // Reset the vendor name if it's not one known to LLVM. This can occur when
  // the triple arch name is set to a value containing hyphens, such as
  // "armv8.2-a". In this case, the vendor is set to "a", which is unknown.
  if (triple.getVendor() == Triple::UnknownVendor)
    triple.setVendor(Triple::VendorType::UnknownVendor);

  opts->Triple = triple.str();

  // Gather features from clang.
  auto featuresOr = getFeaturesFromClang(opts, mcpu);
  if (featuresOr)
    return featuresOr.takeError();

  return TargetInfo(std::move(triple), std::move(opts->CPU),
                    std::move(*featuresOr));
}
