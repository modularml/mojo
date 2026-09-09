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

#include "Init/Init.h"
#include "Support/CommandLine.h"
#include "Support/Configuration.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

namespace {

enum class Property {
  CrashDBPath,
  HandlerPath,
};

struct CLOptions {
  cl::opt<Property> property{
      "get", M::cl::desc("Available properties:"),
      cl::values(
          clEnumValN(Property::CrashDBPath, "crashdb", "Crash database path"),
          clEnumValN(Property::HandlerPath, "crashpad-handler",
                     "Crashpad handler path")),
      llvm::cl::Required};
};

} // namespace

int main(int argc, char **argv) {
  CLOptions clOptions;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Modular Crash Report Path Info Tool");

  auto ctxOr = Init::createContext("crash-report-path-info");
  if (ctxOr.isError()) {
    llvm::errs() << "could not create context: " << ctxOr.getError() << "\n";
    return EXIT_FAILURE;
  }

  auto modularHomeOr = Config::getModularDataFolderPath(/*create=*/false);
  if (modularHomeOr.isError()) {
    llvm::errs() << "could not determine crash path: "
                 << modularHomeOr.getError() << '\n';
    return EXIT_FAILURE;
  }

  switch (clOptions.property) {
  case Property::CrashDBPath: {
    auto path = getCrashDatabasePath(*modularHomeOr);
    llvm::outs() << path.native() << '\n';
    break;
  }
  case Property::HandlerPath:
    if (auto pathOr = getCrashpadHandlerPath((*ctxOr)->get<Config>())) {
      llvm::errs() << "could not determine crashpad handler path: "
                   << pathOr.getError() << '\n';
      return EXIT_FAILURE;
    } else {
      llvm::outs() << pathOr->native() << '\n';
    }
    break;
  }

  return EXIT_SUCCESS;
}
