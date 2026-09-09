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

#include "MojoVariantTypeNameParser.h"
#include <cstring>

namespace M::KGEN::Mojo {

llvm::SmallVector<VariantTypeName>
extractVariantTypeNames(llvm::StringRef typeName) {
  llvm::SmallVector<VariantTypeName> result;

  auto makeEntry = [](llvm::StringRef full) -> VariantTypeName {
    size_t colonPos = full.rfind("::");
    llvm::StringRef tail = colonPos == llvm::StringRef::npos
                               ? full
                               : full.drop_front(colonPos + 2);
    return {std::string(full), std::string(tail)};
  };

  // ── DWARF format ────────────────────────────────────────────────────────
  // Wire format documented in the header; `\22` is LLDB's octal for `"`.
  //
  // Use `find` (not `rfind`) for the closing `\22]`: on aarch64 Linux the
  // symbol has a trailing TypeList with a second identical list, and
  // `rfind` would span both.
  //
  // Once the DWARF opener matches we commit to DWARF and return empty on
  // a malformed close, rather than falling through to REPL form and
  // misreading a stray ` [@...]` further down the symbol.
  {
    constexpr const char *kOpen = R"([@\22)";
    constexpr const char *kClose = R"(\22])";
    constexpr const char *kSep = R"(\22, @\22)";

    size_t start = typeName.find(kOpen);
    if (start != llvm::StringRef::npos) {
      start += strlen(kOpen);
      size_t end = typeName.find(kClose, start);
      if (end == llvm::StringRef::npos || end <= start)
        return result;
      llvm::StringRef params = typeName.slice(start, end);
      while (!params.empty()) {
        llvm::StringRef entry;
        size_t sep = params.find(kSep);
        if (sep == llvm::StringRef::npos) {
          entry = params;
          params = {};
        } else {
          entry = params.slice(0, sep);
          params = params.drop_front(sep + strlen(kSep));
        }
        result.push_back(makeEntry(entry));
      }
      return result;
    }
  }

  // ── REPL format ─────────────────────────────────────────────────────────
  // Wire format in header. Use `find` for the same reason as DWARF above.
  // Entries have `@` sigils between path components (`std::@foo::@Bar`);
  // we strip them so the FQN matches DWARF form.
  {
    size_t start = typeName.find(" [@");
    if (start == llvm::StringRef::npos)
      return result;
    start += 3; // skip " [@"

    size_t end = typeName.find(']', start);
    if (end == llvm::StringRef::npos)
      return result;

    llvm::StringRef params = typeName.slice(start, end);

    // Separator between entries is ", @".
    while (!params.empty()) {
      llvm::StringRef entry;
      size_t sep = params.find(", @");
      if (sep == llvm::StringRef::npos) {
        entry = params;
        params = {};
      } else {
        entry = params.slice(0, sep);
        params = params.drop_front(sep + 3); // skip ", @"
      }

      // Strip the `@` sigils that delimit path components: "std::@foo::@Bar"
      // → "std::foo::Bar".
      std::string full;
      full.reserve(entry.size());
      for (char c : entry)
        if (c != '@')
          full.push_back(c);
      if (!full.empty())
        result.push_back(makeEntry(full));
    }
  }

  return result;
}

} // namespace M::KGEN::Mojo
