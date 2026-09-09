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

#ifndef KGEN_TOOLS_COMPILATION_SERVER_H
#define KGEN_TOOLS_COMPILATION_SERVER_H

#include "mlir/Support/LogicalResult.h"

namespace llvm::lsp {
class JSONTransport;
} // namespace llvm::lsp

namespace M::KGEN {
/// Run the main loop using the given transport.
mlir::LogicalResult runCompilationServer(llvm::lsp::JSONTransport &transport,
                                         bool singleThreaded);
} // namespace M::KGEN

#endif // KGEN_TOOLS_COMPILATION_SERVER_H
