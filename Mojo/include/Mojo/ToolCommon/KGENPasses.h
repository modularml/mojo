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

#ifndef KGEN_TOOLCOMMON_KGENPASSES_H
#define KGEN_TOOLCOMMON_KGENPASSES_H

#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/ADT/DenseStringMap.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/SmallSet.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

class MlirOperation;
class MlirRewriterBase;

namespace mlir {
class ModuleOp;
class OpPassManager;
namespace index {
class IndexDialect;
} // namespace index
namespace LLVM {
class LLVMDialect;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M {
class TargetInfoAttr;
class MDialect;

namespace HLCF {
class HLCFDialect;
} // namespace HLCF

namespace AsyncRT {
class CPUDevice;
} // namespace AsyncRT

namespace KGEN {
class KGENCallOpInterface;
class KGENDialect;
class FuncOp;
class GeneratorOp;
class SymbolConstantAttr;
struct ExportedSymbol;

namespace POP {
class POPDialect;
} // namespace POP

//===----------------------------------------------------------------------===//
// Shared Enums
//===----------------------------------------------------------------------===//

/// Policy for when to update debug info while inlining.
///
/// When compiling with debug info, an op's location is tagged with a source
/// scope (e.g. function scope). This needs to be updated when the op is inlined
/// into another function (with a different scope).
enum class InlinerDebugInfoUpdateTime {
  // Update debug info for each inlined function immediately after it is
  // inlined. More costly, but allows for more optimization while inlining.
  kImmediate,
  // Update debug info after all inlining is done. This requires tagging
  // each inlined scope with an attribute during inlining, and looking for the
  // tag at the end to update debug scopes.
  kDeferred,
  // Never update debug info. This is only legal when compiling without any
  // debug info.
  kNever
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Mojo/KGENPasses.h.inc"

/// Register all passes with default options.
void registerDefaultKGENPasses(const std::string &cacheBaseExtra);

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

/// This struct represents the result of a cross-device compilation, which is a
/// function or closure reference.
struct CrossDeviceFunction {
  /// The compiled cross-device function contents, which can be assembly, object
  /// code, or something else.
  StringAttr contents;
  /// The number of captures that need to be propagated to the cross-device
  /// closure.
  unsigned numCaptures;
  /// A function for populating the captured values opaquely.
  /// FIXME(#22670): The expected API is tightly bound with the GPU module in
  /// the standard library.
  OwningOpRef<Operation *> populateCapturesFn;
};

struct OffloadInfo {
  struct KernelInfo {
    StringAttr name;
    uint64_t kernelId;
    FuncTypeGeneratorType populateFnType;
    llvm::SmallSet<EmitAs, 4> emissionKinds;
    /// The source-level name of the kernel function (e.g. "add_one_kernel"),
    /// used to name the GPU ASM output file when --emit=asm is requested.
    /// Not set when the source name could not be determined.
    std::optional<StringAttr> sourceName;
  };

  struct Group {
    using SymbolInfo = llvm::MapVector<SymbolConstantAttr, KernelInfo>;

    uint64_t numKernels = 0;
    ExportMap exportedSymbols;
    llvm::MapVector<Operation *, SymbolInfo> symbols;
    /// The compile-only emission options string (from `emission_option` on the
    /// `kgen.compile_offload` op). Used to drive `parseEmissionOptions` and
    /// `resetEmissionOptions`.
    std::string emissionOptions;
    /// Extra options forwarded to `CompilationOptions::emissionLinkOptions`
    /// during kernel compilation. Corresponds to `emission_link_option` on the
    /// `kgen.compile_offload` op.
    std::string emissionLinkOptions;
  };

  /// Groups are keyed by the concatenation of `emission_option` and
  /// `emission_link_option` so that kernels with different link options are
  /// compiled separately.
  using EmissionOptionsGroup = llvm::MapVector<std::string, Group>;
  EmissionOptionsGroup groups;
};

struct OffloadCompilationResult {
  OwningOpRef<Operation *> func;
  IntegerAttr numCaptures;
  mlir::DenseI64ArrayAttr captureSizes;
  SymbolConstantAttr populate;
  DenseMap<EmitAs, StringAttr> contents;

  /// Hashed module name based on contents.
  DenseMap<EmitAs, StringAttr> moduleNames;
};

using EmissionOptions = ArrayRef<StringRef>;
/// Function to slice and compile the generator to assembly with the provided
/// input parameters and target. The expected mangled name of the generate is
/// passed to be used as the entry point.
using ElaboratorCompileAsmFn = ErrorOr<CrossDeviceFunction> (*)(
    GeneratorOp, SymbolConstantAttr, StringAttr, const SymbolTable &,
    TargetInfoAttr, EmitAs, EmissionOptions, CompilationOptions,
    ElaborateGeneratorsOptions);

/// Function to prepare slice and compile the generator to its offload target
/// with the provided input parameters and target. The actual compilation
/// will happen later when all generators for the same target are collected and
/// will be compiled together.
/// The expected mangled name of the generate is passed to be used
/// as the entry point.

using ElaboratorCompileOffloadRetType = ErrorOr<DenseMap<
    TargetInfoAttr,
    llvm::DenseMap<std::string, DenseMap<uint64_t, OffloadCompilationResult>>>>;

using ElaboratorCompileOffloadFn = ElaboratorCompileOffloadRetType (*)(
    ModuleOp module,
    llvm::MapVector<TargetInfoAttr, OffloadInfo> &targetOffloadInfos,
    const SymbolTable &, CompilationOptions, ElaborateGeneratorsOptions);

/// Create an instance of the elaborator pass that captures all of the
/// referenced include files.
std::unique_ptr<mlir::Pass>
createElaborateGenerators(TargetInfoAttr target,
                          const ElaborateGeneratorsOptions &elabOpts = {},
                          const CompilationOptions &options = {},
                          ElaboratorCompileAsmFn compileAsmFn = {},
                          ElaboratorCompileOffloadFn compileOffloadFn = {});

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

/// Create an AutomaticInline pass with a function pass pipeline to run.
std::unique_ptr<mlir::Pass> createAutomaticInline(
    const AutomaticInlineOptions &options = {},
    std::function<void(mlir::OpPassManager &)> buildFuncPasses = {});

//===----------------------------------------------------------------------===//
// ReorderParamOps
//===----------------------------------------------------------------------===//

/// Create ReorderParamOps pass with verifier disabled or not.
std::unique_ptr<mlir::Pass> createReorderParamOps(bool disableVerifier);

//===----------------------------------------------------------------------===//
// TargetSpecificLLVMLowering
//===----------------------------------------------------------------------===//
//
// The `createTargetSpecificLLVMLowering()` factory is generated from the
// tablegen pass definition (see `KGENPasses.td`); no manual declaration is
// needed here.

//===----------------------------------------------------------------------===//
// LowerToLLVMPipeline
//===----------------------------------------------------------------------===//
/// Options for the KGEN to LLVM pipeline.
struct LowerToLLVMOptions
    : public mlir::PassPipelineOptions<LowerToLLVMOptions> {
  LowerToLLVMOptions(
      unsigned optLevel = 3,
      DebugInfo::EmissionKind diLevel = DebugInfo::EmissionKind::None,
      std::optional<CompilationOptions::DebugAtLevel> diAtLevel = std::nullopt,
      llvm::dwarf::SourceLanguage diLanguage = llvm::dwarf::DW_LANG_Mojo) {
    optimizationLevel = optLevel;
    debugInfoLevel = diLevel;
    if (diAtLevel)
      debugAtLevel = *diAtLevel;
    debugInfoLanguage = diLanguage;
  }

  Option<unsigned> optimizationLevel{
      *this, "optimization-level",
      llvm::cl::desc("The level of optimization to use"), llvm::cl::init(3)};

  Option<DebugInfo::EmissionKind> debugInfoLevel{
      *this, "debug-level",
      llvm::cl::desc("The level of debug info to use during compilation"),
      llvm::cl::values(
          clEnumValN(DebugInfo::EmissionKind::None, "none",
                     "Disable all debug info."),
          clEnumValN(DebugInfo::EmissionKind::LineTablesOnly, "line-tables",
                     "Only generate debug info for line number tables."),
          clEnumValN(DebugInfo::EmissionKind::Full, "full",
                     "Generate full debug info.")),
      llvm::cl::init(DebugInfo::EmissionKind::None)};

  Option<CompilationOptions::DebugAtLevel> debugAtLevel{
      *this, "debug-at",
      llvm::cl::desc("The abstraction level to generate debug info at"),
      llvm::cl::values(clEnumValN(KGEN::CompilationOptions::kDebugAtLLVM,
                                  "llvm",
                                  "Generate debug info for the LLVM level."))};

  Option<llvm::dwarf::SourceLanguage> debugInfoLanguage{
      *this, "debug-info-language",
      llvm::cl::desc("The DWARF language to specify in the debug info. "
                     "Either `C` or `Mojo`. Defaults to `Mojo`."),
      llvm::cl::values(
          clEnumValN(llvm::dwarf::DW_LANG_C, "C", "C language."),
          clEnumValN(llvm::dwarf::DW_LANG_Mojo, "Mojo", "Mojo language")),
      llvm::cl::init(llvm::dwarf::DW_LANG_Mojo)};

  Option<std::string> globalCtorFnName{
      *this, "global-ctor-fn-name",
      llvm::cl::desc("The name of the global init function in JIT mode."),
      llvm::cl::init("kgenGlobalCtor")};

  Option<std::string> globalDtorFnName{
      *this, "global-dtor-fn-name",
      llvm::cl::desc("The name of the global deinit function in JIT mode"),
      llvm::cl::init("kgenGlobalDtor")};
};

/// Register the lower to LLVM pipeline.
void registerLowerToLLVMPipeline();

/// Utils for emission options.
ErrorOrSuccess parseEmissionOptions(EmissionOptions emissionOptions);
ErrorOrSuccess resetEmissionOptions(EmissionOptions emissionOptions);

} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_KGENPASSES_H
