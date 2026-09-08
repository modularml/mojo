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

#include "Mojo/MojoTooling/DocGen.h"

#include "Config/Version.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Driver/DiagnosticFormat.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;

bool M::generateMojoDocJSON(const std::filesystem::path &resolvedPath,
                            mlir::MLIRContext &context,
                            const DocGenConfig &config, llvm::raw_ostream &os) {
  llvm::SourceMgr sourceManager;
  sourceManager.setDiagHandler(getDiagHandler(config.diagnosticFormat));
  sourceManager.setIncludeDirs(config.includePaths);

  // Add a sentinel buffer as the "main file" so that diagnostics emitted for
  // decls without valid source locations (e.g., from bytecode packages) have a
  // fallback location to reference.
  sourceManager.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer("", ""),
                                   llvm::SMLoc());

  CompilationOptions compilationOptions;
  compilationOptions.warningsAsErrors = config.warningsAsErrors;
  compilationOptions.ignoredDeprecations =
      llvm::to_vector_of<std::string>(config.ignoredDeprecations);
  LIT::ParserConfig parserConfig(&context, compilationOptions);
  parserConfig.diagnoseMissingDocStrings = config.diagnoseMissingDocStrings;
  parserConfig.maxNotesPerDiagnostic = config.maxNotesPerDiagnostic;
  parserConfig.stripFilePrefix = config.stripFilePrefix;
  parserConfig.docsBasePath = config.docsBasePath;

  MojoParserContext parserContext(sourceManager, parserConfig);
  MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(resolvedPath);
  if (!moduleDecl || parserContext.wasErrorEmitted())
    return false;

  std::unique_ptr<PublicDecl> publicDecl = moduleDecl.getDecl();
  if (!publicDecl)
    return false;

  llvm::json::OStream jsonOS(os, /*IndentSize=*/2);
  const char *version = getMojoVersionString();
  jsonOS.value(llvm::json::Object({
      {"decl", publicDecl->toJSON(parserContext)},
      {"version", llvm::formatv("{0}", version).str()},
  }));

  return true;
}
