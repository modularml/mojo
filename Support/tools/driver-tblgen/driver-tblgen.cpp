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

#include "BackendRegistry.h"
#include "GenHelpText.h"
#include "GenManPage.h"
#include "GenMarkdown.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include <cstddef>

using namespace M;

namespace {
/// A command line options parser than can parse and print registered backends.
struct BackendNameParser : public llvm::cl::parser<const Backend *> {
  /// Include the base parser class's constructor.
  using llvm::cl::parser<const Backend *>::parser;

  /// Adds each registered backend name as a parseable option.
  void registerBackends(const BackendRegistry &registry) {
    for (const auto &b : registry.getBackends())
      addLiteralOption(b.name, &b, b.description);
  }

  /// When printing help text, prints each registered backend's name and
  /// description, in alphabetical order.
  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const override {
    BackendNameParser *p = const_cast<BackendNameParser *>(this);
    llvm::array_pod_sort(p->Values.begin(), p->Values.end(),
                         [](const BackendNameParser::OptionInfo *lhs,
                            const BackendNameParser::OptionInfo *rhs) {
                           return lhs->Name.compare(rhs->Name);
                         });
    llvm::cl::parser<const Backend *>::printOptionInfo(opt, globalWidth);
  }
};
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  // Register backends.
  BackendRegistry registry;
  registerGenHelpTextBackend(registry);
  registerGenManPageBackend(registry);
  registerGenMarkdownBackend(registry);

  // Register the backend option and its option parser.
  llvm::cl::opt<const Backend *, false, BackendNameParser> backend(
      llvm::cl::desc("The following generators are available. If none is "
                     "selected, all records are printed:"));
  backend.getParser().registerBackends(registry);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  return llvm::TableGenMain(
      argv[0],
      [&backend](llvm::raw_ostream &os,
                 const llvm::RecordKeeper &records) -> bool {
        // If no backend was selected, print records and exit successfully.
        if (!backend) {
          os << records;
          return false;
        }
        // Otherwise, invoke the selected backend function.
        return backend.getValue()->function(os, records);
      });
}
