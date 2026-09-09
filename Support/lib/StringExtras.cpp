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

#include "Support/StringExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

void M::replaceAll(std::string &str, StringRef oldStr, StringRef newStr) {
  size_t pos = 0;
  size_t oldSize = oldStr.size();

  if (oldSize == 0)
    return;

  while ((pos = str.find(oldStr, pos)) != StringRef::npos) {
    str.replace(pos, oldSize, newStr);
    pos += newStr.size();
  }
}

static std::string prettyPrint(double val, size_t scale = 1,
                               int precision = 2) {
  std::stringstream stream;
  val = val / scale;
  if (std::floor(val) == val)
    stream << static_cast<size_t>(val);
  else
    stream << std::setprecision(precision) << std::fixed << val;
  return stream.str();
}

/// Prints the memory size in a human form with the units.
std::string M::humanMemorySize(size_t size) {
  // These variables do not match our coding style for variable naming (which is
  // camelBack). Since they represent well-known units, ignore them with
  // clang-tidy.
  constexpr size_t KB = 1024;    // NOLINT
  constexpr size_t MB = KB * KB; // NOLINT
  constexpr size_t GB = KB * MB; // NOLINT

  if (size >= GB)
    return prettyPrint(size, GB) + "GB";
  if (size >= MB)
    return prettyPrint(size, MB) + "MB";
  if (size >= KB)
    return prettyPrint(size, KB) + "KB";

  return prettyPrint(size) + "B";
}

/// Checks if the string represents something like True. For example, the
// strings "On" and "1" are considered true. The use case here is to have
// a utility function so that we are forgiving in the input that cannot
// be easily check (e.g. when specifying behavior via an environment variable)
bool M::isTrueLike(StringRef str) {
  return str == "1" || str.equals_insensitive("on") ||
         str.equals_insensitive("true");
}

/// Checks if the string represents something like False. For example, the
// strings "Off" and "0" are considered false. The use case here is to have
// a utility function so that we are forgiving in the input that cannot
// be easily check (e.g. when specifying behavior via an environment variable)
bool M::isFalseLike(StringRef str) {
  return str == "0" || str.equals_insensitive("off") ||
         str.equals_insensitive("false");
}
