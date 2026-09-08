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

#ifndef KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H
#define KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H

#include "lldb/API/LLDB.h"

namespace M::KGEN::Mojo {
/// Register all related `llvm-debug` commands in the given debugger.
void registerLLVMDebugCommands(lldb::SBDebugger debugger);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H
