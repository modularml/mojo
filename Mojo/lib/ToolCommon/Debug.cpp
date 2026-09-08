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
#include "Mojo/ToolCommon/Debug.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace M;
using namespace M::KGEN;

#undef isCurrentDebugTypeLevel

namespace {
static llvm::ManagedStatic<std::vector<std::pair<std::string, unsigned>>>
    currentdebugTypeLevel;

struct debugOnlyOpt {
  void operator=(const std::string &Val) const {
    if (Val.empty())
      return;
    debugFlag = true;
    SmallVector<StringRef, 8> dbgTypeLevels;
    StringRef(Val).split(dbgTypeLevels, ',', -1, false);
    for (StringRef dbgTypeLevel : dbgTypeLevels) {
      SmallVector<StringRef, 2> typeLevel;
      StringRef(dbgTypeLevel).split(typeLevel, ':', -1, false);
      unsigned level = -1U;
      if (typeLevel.size() == 2)
        level = StringRef(typeLevel[1]).getAsInteger(10, level);
      currentdebugTypeLevel->push_back(
          std::make_pair(std::string(dbgTypeLevel), level));
    }
  }
};
} // namespace

namespace M::KGEN {

/// Indicates that debug output is enabled.
bool debugFlag = false;

/// Check if the specified debug type is in the list and if the level specified
/// in macro is smaller than maximum allowed level in the option.
bool isCurrentDebugTypeLevel(const char *debugType, unsigned level) {
  if (currentdebugTypeLevel->empty())
    return true;
  for (auto &d : *currentdebugTypeLevel) {
    if (d.first == debugType && d.second >= level)
      return true;
  }
  return false;
}

} // namespace M::KGEN

#ifndef NDEBUG

static debugOnlyOpt debugOnlyOptLoc;

namespace {
struct CreateDebugOnly {
  static void *call() {
    // The memory is managed by ManagedStatic as a part of debugOnly variable
    // below.
    return new llvm::cl::opt<debugOnlyOpt, true, llvm::cl::parser<std::string>>(
        "kgen-debug-only",
        llvm::cl::desc(
            "Enable debug output for the specified types and maximum level."
            "For example, <pass-name1:level1>,<pass-name2:level2>. To work "
            "correctly multithreading needs to be disabled."),
        llvm::cl::Hidden, llvm::cl::value_desc("debug string"),
        llvm::cl::location(debugOnlyOptLoc), llvm::cl::ValueRequired);
  }
};
} // namespace

static llvm::ManagedStatic<
    llvm::cl::opt<debugOnlyOpt, true, llvm::cl::parser<std::string>>,
    CreateDebugOnly>
    debugOnly;

void M::KGEN::initializeDebugOptions() { *debugOnly; }
#else
void M::KGEN::initializeDebugOptions() {}
#endif // NDEBUG
