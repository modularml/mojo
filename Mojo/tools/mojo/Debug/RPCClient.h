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

#ifndef MOJO_DEBUG_RPC_SERVER_H
#define MOJO_DEBUG_RPC_SERVER_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <optional>

namespace M {
/// Starts an `attach` debug session with an existing RPC debug server.
/// If `dryRun` is specified, then the request payload is printed to the
/// standard output instead.
ErrorOrSuccess invokeAttachRPC(bool dryRun, bool useCudaGdb, bool breakOnLaunch,
                               ArrayRef<int> rpcPorts,
                               const std::optional<StringRef> &pid,
                               const std::optional<StringRef> &processName,
                               ArrayRef<std::string> initCommands);

/// Starts a `launch` debug session with an existing RPC debug server.
/// If `dryRun` is specified, then the request payload is printed to the
/// standard output instead.
ErrorOrSuccess invokeLaunchRPC(bool dryRun, bool useCudaGdb, bool breakOnLaunch,
                               ArrayRef<int> rpcPorts, StringRef target,
                               ArrayRef<std::string> runArgs,
                               StringRef rpcTerminal, bool stopOnEntry,
                               ArrayRef<std::string> initCommands);
} // namespace M

#endif // MOJO_DEBUG_RPC_SERVER_H
