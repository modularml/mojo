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

#include "mojo-precompile.h"
#include "../Common/Compilation.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Cache/CachedTransform.h"
#include "Init/Init.h"
#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/Compiler/Diags.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/Config.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"

#include "Support/Filesystem/Paths.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/xxhash.h"

#include <filesystem>
#include <stack>
#include <tuple>
#include <utility>

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Options includes + OptTable
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Precompile/PrecompileOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct PrecompileOptTable : public llvm::opt::PrecomputedOptTable {
  PrecompileOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

/// This function takes a parsed `lit.package` op and creates  a new
/// `lit.package` op. This new `lit.package` op is one that can be serialized
/// into MLIR bytecode and written to disk, as a `.mojoc` file. The generated
/// package retains the full post-parse IR, including function bodies, in its
/// regions, filtering out ops unneeded once packaged (import ops under a
/// package, doc-string source locations). It is suitable for importing into
/// other Mojo programs, where bodies are materialized lazily on demand.
static std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp>
buildPackageModule(ModuleOp theModule, LIT::PackageOp parsedPackageOp) {
  OwningOpRef<ModuleOp> packageModule =
      ModuleOp::create(parsedPackageOp->getLoc());
  auto b = OpBuilder::atBlockEnd(packageModule->getBody());

  // Clone the relevant operations into the package.
  std::stack<SmartVariant<Operation *, OpBuilder::InsertPoint>> worklist;

  // Clone an op without its regions, and ensure that once that op is finished
  // processing, we reset the OpBuilder's insert point to where it was before we
  // walked the ops inside `op`.
  auto cloneWithoutRegions = [&](auto op) {
    // Save the insert point on the worklist stack first.
    worklist.push(b.saveInsertionPoint());

    auto clonedOp = b.cloneWithoutRegions(op);
    clonedOp.getBodyRegion().push_back(new Block);
    b.setInsertionPointToStart(clonedOp.getBody());
    return clonedOp;
  };

  // Push the ops in `opList` onto the worklist, while preserving their original
  // order.
  auto pushOpsOntoWorklist = [&](auto opList) {
    // Push onto a temporary stack so we can reverse it when we put it onto the
    // worklist - this ensures the ops keep their original order.
    std::stack<Operation *> tmp;
    for (Operation &op : opList)
      tmp.push(&op);

    while (!tmp.empty()) {
      worklist.emplace(tmp.top());
      tmp.pop();
    }
  };

  // Include any generated function thunks at the top-level. These are
  // deduplicated when the package is imported.
  for (auto func : theModule.getOps<LIT::FnOp>()) {
    assert(func.getThunkKeyAttr() && "top-level function must be a thunk");
    pushOpsOntoWorklist(MutableArrayRef(*func));
  }
  for (auto trait : theModule.getOps<LIT::TraitDeclOp>()) {
    if (trait.getClosureSignature().has_value())
      pushOpsOntoWorklist(MutableArrayRef(*trait));
  }

  // Clone the parsed package operation and push its ops onto the worklist.
  LIT::PackageOp thePackage = cloneWithoutRegions(parsedPackageOp);
  pushOpsOntoWorklist(parsedPackageOp.getOps());

  while (!worklist.empty()) {
    auto listFront = worklist.top();
    worklist.pop();
    if (isa<OpBuilder::InsertPoint>(listFront)) {
      b.restoreInsertionPoint(cast<OpBuilder::InsertPoint>(listFront));
      continue;
    }
    TypeSwitch<Operation *>(cast<Operation *>(listFront))
        // Always clone and recurse in the case of a package, module, or struct.
        .Case<LIT::PackageOp, LIT::FileModuleOp, LIT::StructDeclOp>(
            [&](auto op) {
              cloneWithoutRegions(op);
              pushOpsOntoWorklist(op.getOps());
            })
        .Case([&](LIT::UnresolvedImportOp op) {
          // Drop unresolved imports within packages that were used to lazily
          // pull in nested modules. These aren't needed during packaging
          // because everything is recursively resolved.
          if (isa<LIT::PackageOp>(op->getParentOp()))
            return;
          b.clone(*op);
        })
        .Case([&](LIT::ImportOp op) {
          // Drop resolved import ops within packages — they gate access
          // during parsing but are not needed in the packaged bytecode.
          if (isa<LIT::PackageOp>(op->getParentOp()))
            return;
          b.clone(*op);
        })
        // None of the cases matched? Just clone the op directly.
        .Default([&](auto op) { b.clone(*op); });
  }

  // Process the package to strip out various information from the package.
  thePackage.walk([&](LIT::ASTDeclInterface astDeclOp) {
    // Strip out locations from doc strings, these aren't needed anymore.
    auto docAttr = astDeclOp.getDocStringAttr();
    if (docAttr && docAttr.getLocation())
      astDeclOp.setDocStringAttr(LIT::DocStringAttr::get(docAttr.getString()));
  });

  return std::make_pair(std::move(packageModule), thePackage);
}

//===----------------------------------------------------------------------===//
// parsePrecompileArgs
//===----------------------------------------------------------------------===//

namespace {
/// This struct provides an in-memory representation of the arguments passed to
/// the `precompile` subcommand for structured access.
struct PrecompileArgs {
  /// The name of the package being output.
  std::string name;
  /// The package's name while parsing: its source directory's name. Modules
  /// inside the package may import it absolutely under this name, so parsing
  /// must happen under it; `name` is applied when packaging.
  std::string sourceName;
  /// The path to the Mojo package source directory to parse and output as a
  /// package.
  std::string inputPath;
  /// The path to which to output a `.mojoc` file.
  std::string outputPath;
  /// Compilation options common to all Mojo builds.
  CompilationOptions compileOptions;
  /// The MLIR context used for compilation.
  MLIRContext ctx{MLIRContext::Threading::DISABLED};
};
} // namespace

/// Returns whether the given path is an existing directory, or an error if one
/// prevents us from determining anything about the given path.
static ErrorOr<bool> isExistingDirectory(const std::filesystem::path &path) {
  std::error_code ec;
  bool exists = std::filesystem::exists(path, ec);
  if (ec)
    return Error(
        llvm::formatv("could not determine if output path '{0}' exists: {1}",
                      path, ec.message()));
  if (!exists)
    return false;

  bool isDirectory = std::filesystem::is_directory(path, ec);
  if (ec)
    return Error(llvm::formatv(
        "could not determine if output path '{0}' is a directory: {1}", path,
        ec.message()));
  return isDirectory;
}

/// Parse the `precompile` subcommand arguments into a struct.
static ErrorOrSuccess parsePrecompileArgs(const State &state,
                                          const llvm::opt::InputArgList &args,
                                          llvm::SourceMgr &sourceMgr,
                                          PrecompileArgs &pkgArgs) {
  if (!args.hasArg(options::OPT_INPUT))
    return Error("no input directory provided");
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many inputs, expected exactly one");

  // Reject input files that do not appear to be mojo package directories (this
  // includes stdin "-").
  pkgArgs.inputPath = args.getLastArgValue(options::OPT_INPUT).str();
  if (!Filesystem::isMojoSourcePackagePath(pkgArgs.inputPath)) {
    return Error("'" + pkgArgs.inputPath +
                 "' does not correspond to a Mojo package");
  }
  std::string extension = ".mojoc";
  // The package's name comes from its directory, so canonicalize first
  std::error_code ec;
  std::filesystem::path canonicalInput =
      std::filesystem::weakly_canonical(pkgArgs.inputPath, ec);
  if (ec) {
    return Error("cannot canonicalize input path '" + pkgArgs.inputPath +
                 "': " + ec.message());
  }
  std::string inputDirName = canonicalInput.filename().string();
  pkgArgs.sourceName = inputDirName;

  if (args.hasArg(options::OPT_o)) {
    pkgArgs.outputPath = args.getLastArgValue(options::OPT_o);
    if (pkgArgs.outputPath == "-") {
      // If we're outputting to stdout, use the input directory name as the
      // package name.
      pkgArgs.name = inputDirName;
    } else {
      // Otherwise, validate the output path and infer the package name from it.
      std::filesystem::path outputPath(pkgArgs.outputPath);

      // If the user has specified a directory, output an
      // "input-directory-name.mojoc" within that directory.
      ErrorOr<bool> isDirectoryOr = isExistingDirectory(outputPath);
      if (isDirectoryOr.isError())
        return isDirectoryOr.takeError();
      if (*isDirectoryOr) {
        outputPath = outputPath / (inputDirName + extension);
        pkgArgs.outputPath = outputPath;
      }

      if (outputPath.extension() != ".mojoc")
        return Error("output path must have a '.mojoc' extension");

      pkgArgs.name = outputPath.stem().string();
    }
  } else {
    pkgArgs.name = pkgArgs.sourceName;
    pkgArgs.outputPath = pkgArgs.name + extension;
  }

  // Set up the compilation options now, so we can use them as a single source
  // of truth.
  if (auto err = parseCompilationOptions(
          state, args, pkgArgs.compileOptions, sourceMgr, pkgArgs.ctx,
          options::OPT_I, /*optimizationLevelId=*/{}, /*debugLevelId=*/{},
          /*sanitizeId=*/{}, /*sharedLibasan=*/{}, /*externalLibasan=*/{},
          /*bitcodeLibs=*/{}, /*debugInfoLanguageId=*/{}, /*numThreadsId=*/{},
          /*stdLibPath=*/{}, /*loopUnrollingWarnThresholdId=*/{},
          /*elaborationErrorLimitId=*/{},
          /*elaborationErrorIncludePreludeId=*/{},
          /*elaborationErrorVerbose=*/{}, /*elaborationMaxDepth=*/{},
          /*ignoreIncompatiblePrecompiledFileErrorsId=*/
          options::OPT_ignore_incompatible_precompiled_file_errors,
          /*fpModeId=*/{}, options::OPT_ignore_deprecated))
    return err.takeError();

  // Precompiled files are built with the intention of being agnostic, so use
  // conservative compilation settings as a default.
  pkgArgs.compileOptions.debugLevel = CompilationOptions::kFullDebugInfo;
  pkgArgs.compileOptions.optimizationLevel = 0;

  return success();
}

//===----------------------------------------------------------------------===//
// Helper function to write LLVM bitcode module to bytecode attribute
//===----------------------------------------------------------------------===//

/// Reads an LLVM bitcode file and returns it as a DenseResourceElementsAttr.
/// Returns nullptr on failure.
static DenseResourceElementsAttr
writeLLVMBitcodeToDenseAttr(MLIRContext *ctx, StringRef bitcodeFile) {
  // Read the bitcode file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOr =
      llvm::MemoryBuffer::getFile(bitcodeFile);
  if (!bufferOr)
    return {};

  // Get the buffer contents
  llvm::MemoryBuffer &buffer = **bufferOr;
  StringRef data = buffer.getBuffer();

  // Hash the bitcode to generate a unique name
  std::string resourceName =
      "llvm_bitcode_" + llvm::utohexstr(xxh3_64bits(data),
                                        /*LowerCase=*/true, /*Width=*/16);

  // Create the resource attribute
  return createResourceAttr(ctx, ArrayRef<char>(data.data(), data.size()),
                            resourceName);
}

//===----------------------------------------------------------------------===//
// buildPackage
//===----------------------------------------------------------------------===//

/// Given parsed module and package ops, builds a new module and package op. The
/// newly build package op is suitable for serialization as MLIR bytecode; it
/// may be written to a `.mojoc` file that can be deserialized and imported into
/// Mojo programs.
static ErrorOr<OwningOpRef<ModuleOp>>
buildPackage(const PrecompileArgs &precompileArgs, ModuleOp theModule,
             LIT::PackageOp parsedPackageOp, AsyncRT::CPUDevice &cpuDevice) {
  // Add the dependencies of the package to the package itself. Every top-level
  // package other than the one being compiled is an imported precompiled
  // dependency (in the precompile flow, dependencies are always provided as
  // `.mojoc` files), so record each as a link dependency.
  SmallVector<FlatSymbolRefAttr> dependencies;
  for (LIT::PackageOp package : theModule.getOps<LIT::PackageOp>()) {
    // A package is never its own link dependency, under either its parse-time
    // (source directory) name or its output name: a recorded self-dependency
    // dangles once the package is renamed for output.
    if (package == parsedPackageOp ||
        package.getSymName() == precompileArgs.sourceName ||
        package.getSymName() == precompileArgs.name)
      continue;
    dependencies.push_back(FlatSymbolRefAttr::get(package.getSymNameAttr()));
  }
  if (!dependencies.empty()) {
    parsedPackageOp.setDependenciesAttr(
        LinkDependencyArrayAttr::get(theModule.getContext(), dependencies));
  }

  auto [packageModule, thePackage] =
      buildPackageModule(theModule, parsedPackageOp);

  // The package parses under its source directory's name so that absolute
  // self-imports resolve to the package under compilation; the output name
  // applies from here on: to symbol references and the recorded import paths
  // that name the package (both re-resolved when the package is imported,
  // under the output file's name), and to the debug-info source names that
  // snapshot the package lineage.
  if (precompileArgs.name != precompileArgs.sourceName) {
    MLIRContext *ctx = theModule.getContext();
    auto newName = StringAttr::get(ctx, precompileArgs.name);
    auto sourceName = StringAttr::get(ctx, precompileArgs.sourceName);
    // Qualified references rooted at the package resolve through the
    // parser's scoping rather than MLIR symbol tables, so rewrite them
    // directly rather than through SymbolTable::replaceAllSymbolUses.
    //
    // Only a reference's root names a package; its nested components name
    // declarations within one, and a module nested in another package may
    // share the source directory's name. Skip, or the replacer walks into
    // those components and revisits each as a reference rooted at itself.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](SymbolRefAttr ref) -> std::pair<Attribute, WalkResult> {
          if (ref.getRootReference() != sourceName)
            return {ref, WalkResult::skip()};
          return {SymbolRefAttr::get(newName, ref.getNestedReferences()),
                  WalkResult::skip()};
        });
    // Debug-info source names snapshot the symbol lineage as plain strings at
    // parse time, so the package appears in them as a name, not a symbol
    // reference. Rewriting the parentless package root rebuilds every lineage
    // chained from it; these live in scoped locations, hence replaceLocs.
    // Source names for types and parameter values are rendered text, so
    // package-rooted references inside them are substrings: rewrite the
    // root token (its `::` terminator keeps longer names unmatched) wherever
    // it roots a reference. Preceded by `::` it is instead a nested component
    // of a reference rooted elsewhere, and names a module, not the package.
    std::string sourceToken = "@" + precompileArgs.sourceName + "::";
    std::string newToken = "@" + precompileArgs.name + "::";
    auto rewriteNameText = [&](StringAttr str) -> StringAttr {
      StringRef text = str.getValue();
      if (!text.contains(sourceToken))
        return str;
      std::string rewritten;
      for (size_t pos = text.find(sourceToken); pos != StringRef::npos;
           pos = text.find(sourceToken)) {
        StringRef prefix = text.take_front(pos);
        rewritten += prefix;
        rewritten += prefix.ends_with("::") ? sourceToken : newToken;
        text = text.drop_front(pos + sourceToken.size());
      }
      rewritten += text;
      return StringAttr::get(ctx, rewritten);
    };
    replacer.addReplacement([&](DebugInfo::SourceNameAttr attr)
                                -> std::pair<Attribute, WalkResult> {
      bool isPackageRoot =
          attr.getKind() == DebugInfo::SourceNameKind::Package &&
          !attr.getParent() && attr.getName() == sourceName;
      StringAttr name =
          isPackageRoot ? newName : rewriteNameText(attr.getName());
      SmallVector<StringAttr> paramValues;
      for (StringAttr value : attr.getParamValues())
        paramValues.push_back(rewriteNameText(value));
      if (name == attr.getName() &&
          ArrayRef<StringAttr>(paramValues) == attr.getParamValues())
        return {attr, WalkResult::advance()};
      return {DebugInfo::SourceNameAttr::get(
                  name, attr.getParamTypes(), attr.getArgTypes(), paramValues,
                  attr.getParent(), attr.getKind(), attr.getDecorators()),
              WalkResult::advance()};
    });
    // Recorded import paths are raw name components, and any import op form
    // may carry one rooted at the package (a resolved import kept inside a
    // file module, an unresolved import, a wildcard import). All re-resolve
    // under the output file's name when the package is imported.
    replacer.addReplacement(
        [&](LIT::ImportPathAttr path) -> std::pair<Attribute, WalkResult> {
          auto components = path.getComponents();
          if (path.getRelativeLevel() != 0 || components.empty() ||
              components.front() != sourceName)
            return {path, WalkResult::advance()};
          SmallVector<StringAttr> newComponents(components);
          newComponents.front() = newName;
          return {
              LIT::ImportPathAttr::get(ctx, /*relativeLevel=*/0, newComponents),
              WalkResult::advance()};
        });
    replacer.recursivelyReplaceElementsIn(*packageModule,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);
    mlir::SymbolTable::setSymbolName(thePackage, newName);
  }

  // Process bitcode libraries if any were specified
  if (!precompileArgs.compileOptions.bitcodeLibs.empty()) {
    SmallVector<DenseResourceElementsAttr> bitcodeAttrs;

    for (const std::string &bitcodeFile :
         precompileArgs.compileOptions.bitcodeLibs) {
      DenseResourceElementsAttr bitcodeAttr =
          writeLLVMBitcodeToDenseAttr(theModule.getContext(), bitcodeFile);
      if (!bitcodeAttr)
        return Error("failed to load bitcode library: " + bitcodeFile);
      bitcodeAttrs.push_back(bitcodeAttr);
    }

    // Set the bitcode libraries on the package
    thePackage.setExternLLVMBitcodeModulesAttr(
        LIT::DenseResourceElementsArrayAttr::get(theModule.getContext(),
                                                 bitcodeAttrs));
  }

  // Run various check passes now to propagate warnings and errors up to the
  // user.
  KGENCompiler compiler(*theModule.getContext(), precompileArgs.compileOptions);
  if (failed(compiler.runCheckLITPipeline(theModule)))
    return Error("errors occurred during compilation");
  return std::move(packageModule);
}

//===----------------------------------------------------------------------===//
// precompile
//===----------------------------------------------------------------------===//

/// Given the path to a mojo directory, compiles it into a precompiled mojo
/// package op by generating an archive and attaching those bytes to a new
/// top-level `lit.package`, suitable for consumption by other mojo
/// programs.
static int precompile(const State &subcommandState) {
  //===--------------------------------------------------------------------===//
  // Options Parsing
  //===--------------------------------------------------------------------===//

  // Parse command line arguments.
  State state = subcommandState;
  PrecompileOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Precompile/PrecompileOptionsHelpText.inc"
    );
  } else if (args.hasArg(options::OPT_help_hidden)) {
    return state.printHelp(
#include "Precompile/PrecompileOptionsHelpHiddenText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format, options::OPT_disable_warnings,
          options::OPT_werror, options::OPT_wno_error))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  llvm::SourceMgr sourceMgr;
  sourceMgr.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  PrecompileArgs precompileArgs;
  if (auto err = parsePrecompileArgs(state, args, sourceMgr, precompileArgs))
    return state.reportError(err.getError());

  // Create our context (including the cpuDevice).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withCPUDeviceOptions(AsyncRT::CPUDeviceOptions()),
      "precompile");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(precompileArgs.ctx, ctx);

  //===--------------------------------------------------------------------===//
  // Build the package
  //===--------------------------------------------------------------------===//

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(precompileArgs.outputPath, &outputError);
  if (!out)
    return state.reportError(outputError);

  // Parse the input directory as a Mojo package. This returns a module op that
  // wraps the `lit.package` op, which represents the package contents.
  AsyncRT::CPUDevice &cpuDevice = *ctx->get<AsyncRT::CPUDevice>();
  LIT::PackageOp packageOp;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                    &precompileArgs.ctx);
  if (args.hasArg(options::OPT_bitcode_libs)) {
    precompileArgs.compileOptions.bitcodeLibs = llvm::to_vector_of<std::string>(
        args.getAllArgValues(options::OPT_bitcode_libs));
  }

  ScopedMLIRWarningHandler warningHandler(
      &precompileArgs.ctx, precompileArgs.compileOptions.disableWarnings,
      precompileArgs.compileOptions.warningsAsErrors);

  ErrorOr<OwningOpRef<ModuleOp>> module = invokeMojoParser(
      state, args, precompileArgs.compileOptions, &precompileArgs.ctx,
      cpuDevice, options::OPT_diagnose_missing_doc_strings,
      options::OPT_max_notes,
      /*definesId=*/llvm::opt::OptSpecifier(), options::OPT_strip_file_prefix,
      options::OPT_disable_builtins, options::OPT_mojo_search_paths,
      options::OPT_fixit, options::OPT_export_fixit,
      /*timingScope=*/nullptr,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        OwningOpRef<ModuleOp> moduleOp;
        std::tie(moduleOp, packageOp) = LIT::importMojoPackage(
            ctx, precompileArgs.inputPath, precompileArgs.sourceName, sourceMgr,
            parserConfig, ts);
        return moduleOp;
      });
  if (failed(module))
    return state.reportError(module.getError());

  if (!module.get()->getOperation()) {
    // Only --experimental-fixit returns a null module (after applying fixes).
    // --experimental-export-fixit continues normal execution after writing
    // YAML.
    assert(args.hasArg(options::OPT_fixit));
    return EXIT_SUCCESS;
  }

  // Build a new package op based off of the parsed package op. This new op is
  // suitable for serialization as MLIR bytecode.
  auto builtOrErr =
      buildPackage(precompileArgs, **module, packageOp, cpuDevice);
  if (failed(builtOrErr))
    return state.reportError(builtOrErr.getError());
  OwningOpRef<ModuleOp> builtPackageModule = builtOrErr.takeValue();

  // Write the new package op as serialized bytecode to the output file.
  if (failed(writePrecompiledFile(&**builtPackageModule, out->os())))
    return state.reportError("failed to write package bytecode to a file");

  out->keep();

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Check if any warnings were promoted to errors via -Werror.
  return warningHandler.wasErrorEmitted() ? EXIT_FAILURE : EXIT_SUCCESS;
}

void M::registerPrecompileSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("precompile", precompile);
  registry.addCallback("package", precompile, "precompile");
}
