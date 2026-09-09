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

#ifndef KGEN_LIB_MOJO_LSP_LSPSERVER_H
#define KGEN_LIB_MOJO_LSP_LSPSERVER_H

#include "Mojo/Support/CompilerProfiling.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace llvm::lsp {
class JSONTransport;
} // namespace llvm::lsp

namespace M::AsyncRT {
class WorkQueue;
} // namespace M::AsyncRT

namespace M::KGEN::LIT {
/// Run the main loop using the given transport.
mlir::LogicalResult
runMojoLSPServer(llvm::lsp::JSONTransport &transport, bool singleThreaded,
                 bool waitOnShutdown, ArrayRef<std::string> includeDirs,
                 std::unique_ptr<KGEN::TraceProfiler> profiler,
                 bool checkDocstringCodeBlocks = false);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJO_LSP_LSPSERVER_H
