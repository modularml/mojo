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

#ifndef KGEN_KGENDIALECT_COMPILATIONCONTEXT_H
#define KGEN_KGENDIALECT_COMPILATIONCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

namespace M::KGEN {

/// Represents kernel compilation options that is used to set mojo parameters
/// during JITing.
struct CompilationContext {
  llvm::DenseMap<llvm::StringRef, std::variant<bool, int, std::string>>
      mojoDefines;

  /// Print the compilation config to the output stream.
  void print(llvm::raw_ostream &os) const;
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_COMPILATIONCONTEXT_H
