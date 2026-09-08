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

#include "Mojo/KGENDialect/KGENCompilationContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace M::KGEN {

void CompilationContext::print(llvm::raw_ostream &os) const {
  // Copy into a sorted vector so the cache key is deterministic regardless of
  // DenseMap's internal iteration order.
  llvm::SmallVector<
      std::pair<llvm::StringRef, std::variant<bool, int, std::string>>>
      sortedDefines(mojoDefines.begin(), mojoDefines.end());
  llvm::sort(sortedDefines, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (const auto &entry : sortedDefines) {
    os << entry.first << '=';
    std::visit([&](const auto &v) { os << v << ';'; }, entry.second);
  }
}

} // namespace M::KGEN
