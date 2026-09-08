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

#include "Mojo/Support/NameMangling.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"
#include <limits>

using namespace M;
using namespace KGEN;

/// Return whether the character is valid. Alnum, underscore, and period
/// characters are valid.
static constexpr bool isValid(char c) { return c == '_' || std::isalnum(c); }

/// Produce an array of all the valid characters. This array will be used to
/// encode the unsupported characters.
static constexpr auto produceCipher() {
  // alnum + underscore is twice the alphabet, each digit, and the underscore.
  constexpr size_t size = 26 * 2 + 10 + 1;
  std::array<char, size> cipher = {};
  unsigned i = 0;
  auto fill = [&](char lb, char ub) {
    for (char c = lb; c <= ub; ++c)
      cipher[i++] = c;
  };
  fill('A', 'Z');
  fill('a', 'z');
  fill('0', '9');
  cipher[i] = '_';
  return cipher;
}

namespace {
/// `std::pair<char, char>` is not constexpr apparently.
struct TwoChars {
  char d0, d1;
};
} // namespace

/// Encode each invalid character as a pair of valid characters.
static constexpr auto produceEncoding() {
  auto cipher = produceCipher();
  static_assert(cipher.size() * cipher.size() >= 256,
                "not enough valid characters");
  std::array<TwoChars, 256> encoding = {};
  for (int c = 0; c < 256; ++c)
    encoding[c] = {cipher[c % cipher.size()], cipher[c / cipher.size()]};
  return encoding;
}

/// Scan `name` and replace each contiguous run of invalid characters with a
/// single '_'.  When `appendEncoding` is true, all invalid characters are also
/// tallied and appended at the end as two-character codes (using the cipher
/// above), so that distinct inputs with different invalid characters can never
/// collide.  When `appendEncoding` is false the replacement is purely
/// positional: each run becomes exactly one '_' and nothing is appended.
static SmallString<1024> replaceInvalidCharacter(StringRef name,
                                                 bool appendEncoding) {
  SmallVector<char, 256> invalid;
  if (appendEncoding)
    invalid.reserve(name.size());
  SmallString<1024> result;
  result.reserve(appendEncoding ? name.size() * 3 : name.size());
  bool carryingInvalid = false;
  for (char c : name) {
    if (isValid(c)) {
      // If the last character was invalid, push an underscore.
      if (carryingInvalid) {
        carryingInvalid = false;
        result.push_back('_');
      }
      // Push the valid character.
      result.push_back(c);
      continue;
    }
    if (appendEncoding)
      invalid.push_back(c);
    carryingInvalid = true;
  }
  if (!appendEncoding || invalid.empty())
    return result;
  static constexpr auto encoding = produceEncoding();
  for (char c : invalid) {
    auto [d0, d1] = encoding[c];
    result.push_back(d0);
    result.push_back(d1);
  }
  return result;
}

/// Shared body for sanitizeSymbolToAlnum / sanitizeSymbolToUnderscores.
static StringAttr sanitizeSymbolImpl(StringAttr name, size_t charToKeep,
                                     bool appendEncoding) {
  SmallString<1024> result;
  if (name.size() > charToKeep) {
    std::string hash = llvm::utohexstr(llvm::xxh3_64bits(name),
                                       /*LowerCase=*/true, /*Width=*/16);
    result = replaceInvalidCharacter(name.strref().take_front(charToKeep),
                                     appendEncoding);
    result += "_";
    result += hash;
  } else {
    result = replaceInvalidCharacter(name, appendEncoding);
  }

  // Prefix with '_' if the result starts with a digit, which is not a valid
  // identifier start in GPU assemblers (e.g. PTX).
  if (!result.empty() && llvm::isDigit(result.front()))
    result.insert(result.begin(), '_');

  return StringAttr::get(name.getContext(), result);
}

StringAttr KGEN::sanitizeSymbolToAlnum(StringAttr name, size_t charToKeep) {
  VerboseCompilerTimeTraceScope traceScope("sanitizeSymbolToAlnum",
                                           [name] { return name.str(); });
  return sanitizeSymbolImpl(name, charToKeep, /*appendEncoding=*/true);
}

/// Like sanitizeSymbolToAlnum but replaces every run of invalid characters with
/// a single '_' without appending their encoded forms.  This produces cleaner
/// PTX names when the source string uses separator characters (e.g. dots) that
/// are meaningful to humans but irrelevant after sanitisation.  Long-name
/// hashing and digit-start fixup are preserved.
StringAttr KGEN::sanitizeSymbolToUnderscores(StringAttr name,
                                             size_t charToKeep) {
  VerboseCompilerTimeTraceScope traceScope("sanitizeSymbolToUnderscores",
                                           [name] { return name.str(); });
  return sanitizeSymbolImpl(name, charToKeep, /*appendEncoding=*/false);
}

/// Append a uniqueness suffix to @p userName to prevent symbol collisions
/// between structurally different instantiations sharing the same @__name
/// prefix. The suffix is "_" + 8 lowercase hex chars of
/// xxh3_64(symName + funcTypeStr).
///
/// \param userName     The sanitized @__name prefix string.
/// \param symName      The auto-mangled wrapper symbol name, used as a hash
///                     input to disambiguate instantiations that share a
///                     prefix.
/// \param funcTypeStr  The printed MLIR function type, used as a hash input to
///                     disambiguate closures capturing different types.
StringAttr KGEN::appendAutoMangledSuffix(StringAttr userName, StringRef symName,
                                         StringRef funcTypeStr) {
  // Hash the auto-mangled sym_name concatenated with the printed function type.
  // symName encodes generator parameter values; funcTypeStr encodes captured
  // variable types, disambiguating instantiations whose parameters are
  // identical but whose captured types differ (e.g. uint32 vs uint64 shape
  // variants of the same closure).
  SmallString<512> toHash;
  toHash += symName;
  toHash += funcTypeStr;
  uint32_t hash = static_cast<uint32_t>(llvm::xxh3_64bits(toHash));
  std::string suffix = llvm::utohexstr(hash, /*LowerCase=*/true, /*Width=*/8);

  SmallString<128> result(userName.getValue());
  result += "_";
  result += suffix;
  return StringAttr::get(userName.getContext(), result);
}
