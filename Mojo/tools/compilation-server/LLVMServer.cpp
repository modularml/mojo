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

#include "LLVMServer.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Init/Init.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/Context.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace mlir;
namespace CSP = M::KGEN::CSP;
using namespace CSP;

//===----------------------------------------------------------------------===//
// LLVMServer::Impl
//===----------------------------------------------------------------------===//

struct LLVMServer::Impl {
  Impl(ContextRef ctx) : mlirCtx(), globalCtx(ctx.copy()) {
    initMLIRContext(mlirCtx);
  }

private:
  void initMLIRContext(MLIRContext &ctx) {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    registerKGENToLLVMTranslation(registry);
    registerContext(registry, globalCtx, /*enableThreadPool=*/true);
    mlirCtx.appendDialectRegistry(registry);
    mlirCtx.allowUnregisteredDialects(true);
  }

public:
  /// MLIR Context
  MLIRContext mlirCtx;
  /// Global context
  ContextRef globalCtx;
};

//===----------------------------------------------------------------------===//
// LLVMServer
//===----------------------------------------------------------------------===//

LLVMServer::LLVMServer(std::unique_ptr<Impl> &&impl) : impl(std::move(impl)) {}
LLVMServer::~LLVMServer() = default;

ErrorOr<std::unique_ptr<LLVMServer>> LLVMServer::create(bool singleThreaded) {
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "compilation-server", Init::Options().withCPUDeviceOptions(
                                AsyncRT::CPUDeviceOptions()
                                    .withSingleThreaded(singleThreaded)
                                    .withMainWillNotDonate()));
  if (ctxOr.isError())
    return ctxOr.takeError();
  auto impl = std::make_unique<Impl>(ctxOr->copy());
  return std::unique_ptr<LLVMServer>(new LLVMServer(std::move(impl)));
}

std::string LLVMServer::echoMLIR(mlir::StringRef module) {
  // Parse MLIR module into ModuleOp
  OwningOpRef<ModuleOp> moduleOp =
      parseSourceString<ModuleOp>(module, mlir::ParserConfig(&impl->mlirCtx));

  if (!moduleOp)
    return "Error: cannot parse MLIR module";

  // Print MLIR module
  std::string str;
  llvm::raw_string_ostream strStream(str);
  strStream << *moduleOp;
  return str;
}

std::string LLVMServer::emitArchive(const EmitArchiveParams &params) {
  // Parse MLIR module into ModuleOp.
  OwningOpRef<ModuleOp> moduleOp = parseSourceString<ModuleOp>(
      params.module, mlir::ParserConfig(&impl->mlirCtx));
  if (!moduleOp)
    return "Error: cannot parse MLIR module";

  // Create object compiler.
  auto compilerOr =
      ObjectCompiler::create(kMojoCacheBaseDirName, params.compilationOptions,
                             params.isJIT, impl->mlirCtx);
  if (failed(compilerOr))
    return "Error: cannot create object compiler";
  ObjectCompiler &objCompiler = **compilerOr;

  // Emit archive.
  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(std::move(moduleOp));
  if (failed(archiveOr))
    return "Error: cannot execute emitArchive";

  // Return emitted archive encoded as text string.
  BufferRef archive = archiveOr.takeValue();
  StringRef buffer = archive->getBuffer();

  return llvm::encodeBase64(buffer);
}
