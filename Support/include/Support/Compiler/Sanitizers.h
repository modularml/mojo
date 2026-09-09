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

#ifndef SUPPORT_COMPILER_SANITIZERS_H
#define SUPPORT_COMPILER_SANITIZERS_H

#include "llvm/Support/raw_ostream.h"

namespace M {
/// The sanitizers enabled for the compilation.
class Sanitizers {
public:
  /// The various sanitizers that can be enabled.
  enum SanitizerKind { kAddress, kThread };

  Sanitizers(unsigned sanitizerMask = 0) : sanitizerMask(sanitizerMask) {}

  /// Check if the given sanitizer is enabled.
  bool has(SanitizerKind sanitizer) const {
    return sanitizerMask & (1 << sanitizer);
  }

  /// Enable the given sanitizer.
  void enable(SanitizerKind sanitizer) { sanitizerMask |= (1 << sanitizer); }

  /// Returns if any sanitizer is enabled.
  operator bool() const { return sanitizerMask != 0; }

  /// Print the active sanitizers to `os`.
  void print(llvm::raw_ostream &os) const {
    if (has(Sanitizers::kAddress))
      os << " address";
    if (has(Sanitizers::kThread))
      os << " thread";
  }

private:
  unsigned sanitizerMask;
};
} // namespace M

#endif // SUPPORT_COMPILER_SANITIZERS_H
