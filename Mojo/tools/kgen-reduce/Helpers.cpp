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

#include "Helpers.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>

using namespace M;

uint64_t M::getCurTimeMs() {
  using namespace std::chrono;
  auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}

std::string M::getTempFileName() {
  return ("kgen-reduce." + Twine(getCurTimeMs())).str();
}

ErrorOr<std::unique_ptr<llvm::ToolOutputFile>>
M::getTempFile(ModuleOp module, const Twine &fileName, StringRef pipeline) {
  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile((fileName + ".mlirbc").str());
  if (!output)
    return Error(err);
  mlir::BytecodeWriterConfig config("kgen-reduce");
  config.attachResourcePrinter("mlir_reproducer",
                               [&](Operation *op, mlir::AsmResourceBuilder &b) {
                                 b.buildString("pipeline", pipeline);
                                 b.buildBool("disable_threading", true);
                                 b.buildBool("verify_each", true);
                               });
  if (failed(mlir::writeBytecodeToFile(module, output->os(), config)))
    return Error("failed to write bytecode");
  return std::move(output);
}

ErrorOrSuccess M::stashFile(ModuleOp module, const Twine &fileName,
                            StringRef pipeline) {
  auto err = getTempFile(module, fileName, pipeline);
  if (err.isError())
    return err.takeError();
  err.takeValue()->keep();
  return success();
}

void M::unkeepToolOutputFile(llvm::ToolOutputFile &file) {
  // HACK: Access private members of the type to reset the flag.
  struct DirtyHack {
    struct CleanupInstaller {
      std::string filename;
      bool keep;
    } installer;

    std::optional<llvm::raw_fd_ostream> osHolder;
    llvm::raw_fd_ostream *os;
  };
  ((DirtyHack *)&file)->installer.keep = false;
}

bool M::isStubbed(Region &region) {
  return isa<KGEN::UnreachableOp>(region.front().front());
}

void M::stubRegion(Region &region, Region &owner) {
  owner.takeBody(region);

  // Create a new block with the same argument kinds.
  region.push_back(new Block);
  for (BlockArgument arg : owner.getArguments())
    region.addArgument(arg.getType(), arg.getLoc());

  // Stub the function with an unreachable.
  OpBuilder b(&region.front(), region.front().begin());
  KGEN::UnreachableOp::create(b, region.getLoc());
}
