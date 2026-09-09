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

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H

#include "llvm/Support/JSON.h"

namespace llvm::json {

/// Function similar to the typical `llvm::json::parse`, but that can operate
/// directly on a `Value` object.
template <typename T>
llvm::Expected<T> parse(const llvm::json::Value &json) {
  llvm::json::Path::Root root("");
  T result;
  if (fromJSON(json, result, root))
    return std::move(result);
  return root.getError();
}

} // namespace llvm::json

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H
