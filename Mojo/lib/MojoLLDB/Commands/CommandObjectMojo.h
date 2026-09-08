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

#ifndef KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H
#define KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H

#include "Support/Context.h"
#include "lldb/API/LLDB.h"

namespace M::KGEN::Mojo {
using GetContextFn = M::ContextRef (*)();

/// Register all related `mojo` commands in the given debugger.
void registerMojoCommands(lldb::SBDebugger debugger, GetContextFn getContext);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H
