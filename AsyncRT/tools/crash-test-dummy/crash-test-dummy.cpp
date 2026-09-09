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
#include "Support/CrashReporting/CrashReporting.h"

using namespace M;

namespace {

struct CLOptions {
  cl::opt<bool> simulate{
      "simulate", M::cl::desc("Simulate crash rather than actually crashing")};
};

} // namespace

int main(int argc, char **argv) {
  CLOptions clOptions;
  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Crash Test Dummy");

  // TODO: Remove when https://github.com/bazelbuild/bazel/pull/26887 is fixed.
  setenv("MODULAR_CRASH_REPORTING_ENABLED", "true", 1);

  auto ctxOr = Init::createContext("crash-test-dummy");
  if (ctxOr.isError()) {
    llvm::errs() << "could not create context: " << ctxOr.getError() << "\n";
    return EXIT_FAILURE;
  }

  if (clOptions.simulate)
    generateNonFatalDump();
  else
    std::abort();

  return EXIT_SUCCESS;
}
