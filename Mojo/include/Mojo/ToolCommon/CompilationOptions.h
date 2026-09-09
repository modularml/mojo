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

#ifndef KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
#define KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#if MLRT_ACCELERATOR_SUPPORT
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#endif
#include "Support/Compiler/Sanitizers.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/TargetParser/Host.h"
#include <cstddef>

namespace M::KGEN {

/// Floating-point mode (`-fp-mode`). `contract` (FMA fusion of `a + b*c`) is on
/// by default.
struct FpMode {
  bool contract = true;
};

enum class FpModeItemStatus { Ok, UnknownFeature, InvalidValue };

/// Parses one `-fp-mode` item of the form `contract=fast|off` into `mode`.
inline FpModeItemStatus parseFpModeItem(llvm::StringRef item, FpMode &mode) {
  auto [feature, value] = item.split('=');
  if (feature != "contract")
    return FpModeItemStatus::UnknownFeature;
  if (value == "fast" || value == "off") {
    mode.contract = value == "fast";
    return FpModeItemStatus::Ok;
  }
  return FpModeItemStatus::InvalidValue;
}

/// Splits the fp-mode items (e.g. `contract=off`) out of the comma-separated
/// emission-option list `options`, applying them to `mode`. Non-fp-mode items
/// are rejoined into `rest`. Returns the offending item if an fp-mode feature
/// carries an invalid value; nullopt otherwise.
inline std::optional<std::string>
splitFpModeEmissionOptions(llvm::StringRef options, FpMode &mode,
                           std::string &rest) {
  llvm::SmallVector<llvm::StringRef> items;
  options.split(items, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::SmallVector<llvm::StringRef> kept;
  for (llvm::StringRef item : items) {
    FpMode trial = mode;
    switch (parseFpModeItem(item, trial)) {
    case FpModeItemStatus::Ok:
      mode = trial;
      break;
    case FpModeItemStatus::UnknownFeature:
      kept.push_back(item);
      break;
    case FpModeItemStatus::InvalidValue:
      return item.str();
    }
  }
  rest = llvm::join(kept, ",");
  return std::nullopt;
}

/// Returns true if `item` is a `target-abi=...` emission option. KGEN
/// intercepts this itself, rather than registering a global cl option, since
/// it must be applied before the TargetMachine is created.
inline bool isTargetABIEmissionOption(llvm::StringRef item) {
  return item.split('=').first == "target-abi";
}

/// Applies a `target-abi=<value>` item in `items` to `targetABI`, forwarding
/// the value verbatim, matching LLVM's own `-target-abi` flag semantics.
inline void applyTargetABIEmissionOptions(llvm::ArrayRef<llvm::StringRef> items,
                                          std::string &targetABI) {
  for (llvm::StringRef item : items) {
    auto [key, value] = item.split('=');
    if (key == "target-abi")
      targetABI = value.str();
  }
}

/// This class provides a set of options used to control the compilation of
/// KGEN modules.
class CompilationOptions {
public:
  /// The debug info level to use when compiling.
  enum DebugInfoLevel {
    /// Disable debug info generation.
    kNoDebug,

    /// Generate synthetic debug info.
    kSynthetic,

    /// Emit only debug info necessary for generating line number tables.
    kLineTablesOnly,

    /// Generate complete debug info.
    kFullDebugInfo,
  };

  // The language to specify in the debug info.
  enum DebugInfoLanguage {
    kLangC = llvm::dwarf::DW_LANG_C,
    kLangMojo = llvm::dwarf::DW_LANG_Mojo,
  };

  /// The compilation abstraction level to generate debug info for, used in
  /// tandem with DebugInfoLevel.
  enum DebugAtLevel {
    kDebugUnset,
    /// Generate debug info for the LLVM output.
    kDebugAtLLVM
  };

  enum ErrorVerboseLevel { kNoParams, kSimpleParams, kAllParams };

  CompilationOptions(
      unsigned optimizationLevel = 3, DebugInfoLevel debugLevel = kNoDebug,
      std::optional<DebugAtLevel> debugAtLevel = std::nullopt,
      Sanitizers sanitizers = Sanitizers(),
      std::string targetTriple = llvm::sys::getDefaultTargetTriple(),
      std::string targetCpu = llvm::sys::getHostCPUName().str(),
      std::string targetFeatures = getHostCPUFeatures(),
      std::string targetAccelerator = "", int elaborationErrorLimit = 20,
      bool elaborationErrorIncludePrelude = false,
      ErrorVerboseLevel elaborationErrorVerbose = kSimpleParams,
      unsigned elaborationMaxDepth = std::numeric_limits<unsigned>::max(),
      DebugInfoLanguage debugInfoLanguage = kLangMojo,
      std::string searchPaths = "",
      SmallVector<std::string> extraSearchPaths = {});

  /// Return the corresponding codegen optimization level for the current option
  /// set.
  llvm::CodeGenOptLevel getCodeGenOptLevel() const;

  /// Return the corresponding debuginfo emission level for the current option
  /// set.
  DebugInfo::EmissionKind getDIEmissionKind() const;

  /// Return the debug info level to use when parsing an input file.
  DebugInfoLevel getDebugInfoLevelForInput() const {
    return debugAtLevel ? kNoDebug : debugLevel;
  }

  /// Print the compilation options to the given stream.
  void print(raw_ostream &os) const;

  /// Parse command line defines, adding defaults based on compilation options.
  ErrorOr<EnvAttr> parseDefinesWithDefaults(MLIRContext *ctx,
                                            ArrayRef<std::string> defines);

  /// Get debugLevel as a string that matches how EnvAttr is being set.
  StringRef getDebugLevelString() const;

  /// Save temporary files to a file with the given prefix.
  void setSaveTemps(std::string prefix) { saveTempsPrefix = prefix; }

  unsigned optimizationLevel = 3;
  FpMode fpMode;
  DebugInfoLevel debugLevel = kNoDebug;
  std::optional<DebugAtLevel> debugAtLevel;
  Sanitizers sanitizers = Sanitizers();
  bool sharedLibasan = false;
  std::string externalLibasan = {};
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string targetCpu = llvm::sys::getHostCPUName().str();
  std::string targetFeatures = getHostCPUFeatures();
  /// User-supplied target ABI name (`--target-abi`, or a `#kgen.target`'s
  /// `abi` field): a target-identity property shared by every kernel
  /// compiled for that target. Forwarded to the TargetMachine (see
  /// `targetABI` below) and recorded as a `target-abi` LLVM module flag,
  /// mirroring clang's `-target-abi`.
  std::string targetAbi;
  std::string targetDataLayout;
  /// Target ABI name forwarded to the TargetMachine (MCOptions.ABIName),
  /// taking precedence over `targetAbi` above when both are set. Unlike
  /// `targetAbi`, this is a per-kernel override (e.g. NVPTX's "shortptr"
  /// ABI) sourced from a `target-abi` emission option, reset before each
  /// compile rather than carried on the target's identity.
  std::string targetABI;
  std::optional<llvm::CodeModel::Model> mcmodel;
  std::optional<uint64_t> largeDataThreshold;
  int64_t loopUnrollingWarnThreshold = 65536;

  std::string targetAccelerator;
  bool isCrossCompilation = false;
  bool useParametricInterpreter = false;

  llvm::Reloc::Model relocModel = llvm::Reloc::Model::PIC_;
  DebugInfoLanguage debugInfoLanguage = kLangMojo;

  std::string saveTempsPrefix;
  /// When set, offload kernel output files are written alongside the host
  /// output.  Each file is named <prefix>_<kernel_name><ext>.
  /// The extension is target-specific for ASM emission,
  /// or target-qualified .ll for LLVM IR emission (controlled by
  /// offloadOutputKind). Colliding names (same kernel, multiple instantiations)
  /// get _1/_2/...
  std::string offloadOutputPrefix;
  /// Selects the offload kernel file format written when offloadOutputPrefix is
  /// set. EmitAs::ASM  → target-specific assembly. EmitAs::LLVM → LLVM IR
  /// text (.ll) for all targets.
  /// --emit=asm and --emit=llvm are mutually exclusive, so only one value is
  /// ever active at a time.
  EmitAs offloadOutputKind = EmitAs::ASM;
  std::string searchPaths;
  SmallVector<std::string> extraSearchPaths;

  // File paths to external bitcode libraries specified via command line.
  SmallVector<std::string> bitcodeLibs;

  bool verboseOutput = false;

  int elaborationErrorLimit = 20;

  bool elaborationErrorIncludePrelude = false;

  ErrorVerboseLevel elaborationErrorVerbose = kSimpleParams;

  unsigned elaborationMaxDepth = std::numeric_limits<unsigned>::max();

  // HACK: to disable llvm splitting for some cases.
  // - mojo REPL (#35345)
  // - graph compiler's compilation path where heuristics is needed for
  // performance.
  // - ...
  bool enableLLVMPerFunctionSplitting = false;
  bool enableParallelLLC = true;

  std::string emissionOptions;

  /// Extra options forwarded from `kgen.compile_offload`'s
  /// `emission_link_option` attribute.
  /// A string directly passed to linker if compile offload
  /// runs one in the pipeline.
  std::string emissionLinkOptions;

  // Maximum number of threads to be used by AsyncRT. 0 means use all available.
  size_t numThreads = 0;

  bool disableWarnings = false;
  bool warningsAsErrors = false;
  bool warnOnUnstableAPIs = false;
  bool ignoreIncompatiblePrecompiledFileErrors = false;

  /// Qualified names (e.g. `Foo.bar`, or just `some_fn` for a top-level
  /// declaration) of `@deprecated` declarations whose deprecation warning
  /// should be suppressed, set via `--ignore-deprecated`.
  SmallVector<std::string> ignoredDeprecations;

  // Extra handle name to separate cache base between mojo, kgen, kgen-opt,
  // to avoid internal test cache pruning races caused by different binary IDs.
  // Set default value to be "mojo", will be overwritten with other tool names
  // if needed.
  std::string cacheBaseExtra = "mojo";

  void setDefaultCPU();
};

// Return true if target triple is `air64-`
bool isMetalTriple(const llvm::Triple &triple);

// Return true if `triple` is a registered GPU target.
bool isGPUTriple(const llvm::Triple &triple);

// Whether the standalone module should export all symbols (GPU targets do, so
// offloaded kernels resolve by symbol at runtime).
bool overrideExported(const llvm::Triple &triple);
bool overrideExported(const CompilationOptions &options);

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
