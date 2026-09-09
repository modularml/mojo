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

#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_FUNCTIONSTATS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct FunctionStatsPass : impl::FunctionStatsBase<FunctionStatsPass> {
  void runOnOperation() override {
    struct FunctionSummary {
      StringRef name;
      Location loc;
      size_t opCount;

      std::string opCountStr;
    };

    SmallVector<FunctionSummary> stats;
    auto checkFunc = [&](mlir::FunctionOpInterface func) {
      FunctionSummary result{func.getName(), func.getLoc(), 0, ""};
      func.walk([&](Operation *op) { ++result.opCount; });
      stats.push_back(std::move(result));
      return WalkResult::skip();
    };

    getOperation().walk<mlir::WalkOrder::PreOrder>(checkFunc);

    // Sort by most number of ops.
    std::stable_sort(stats.begin(), stats.end(), [&](auto &lhs, auto &rhs) {
      return lhs.opCount > rhs.opCount;
    });

    // Print <opcount> | trunc(name) | file_loc
    // Compute the maximum required padding for opcount.
    size_t maxOpCountSize = 0, maxNameLen = 0;
    constexpr size_t nameBound = 128;
    for (FunctionSummary &f : stats) {
      f.opCountStr = Twine(f.opCount).str();
      maxOpCountSize = std::max(maxOpCountSize, f.opCountStr.size());
      maxNameLen = std::min(std::max(maxNameLen, f.name.size()), nameBound);
    }

    llvm::errs() << "\n"
                 << std::string(50, '=') << " Function Stats "
                 << std::string(50, '=') << "\n\n";
    for (FunctionSummary &f : stats) {
      llvm::errs() << std::string(1 + (maxOpCountSize - f.opCountStr.size()),
                                  ' ')
                   << f.opCountStr << " | ";
      llvm::errs() << '@';
      if (f.name.size() > maxNameLen)
        llvm::errs() << f.name.substr(0, maxNameLen);
      else
        llvm::errs() << f.name << std::string(maxNameLen - f.name.size(), ' ');
      llvm::errs() << " | ";
      if (auto loc = f.loc->findInstanceOf<mlir::FileLineColLoc>()) {
        llvm::errs() << loc.getFilename().strref() << ':' << loc.getLine()
                     << ':' << loc.getColumn();
      } else {
        llvm::errs() << loc;
      }
      llvm::errs() << "\n";
    }

    llvm::errs() << "\n" << std::string(116, '=') << "\n\n";

    markAllAnalysesPreserved();
  }
};
} // namespace
