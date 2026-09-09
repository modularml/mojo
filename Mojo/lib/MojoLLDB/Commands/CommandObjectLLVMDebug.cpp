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

#include "CommandObjectLLVMDebug.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "lldb/Target/Target.h"
#include "llvm/Support/Debug.h"

using namespace M;
using namespace lldb;

namespace {
//===----------------------------------------------------------------------===//
// CommandLLVMDebugEnable: llvm-debug enable [types]
//
// E.g., llvm-debug enable modular_memcpy lit-parameter-evaluator
//===----------------------------------------------------------------------===//
class CommandLLVMDebugEnable : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override;
};

//===----------------------------------------------------------------------===//
// CommandLLVMDebugDisable: llvm-debug disable
//===----------------------------------------------------------------------===//
class CommandLLVMDebugDisable : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override;
};
} // namespace

//===----------------------------------------------------------------------===//
// CommandLLVMDebugEnable: llvm-debug enable [types]
//
// E.g., llvm-debug enable modular_memcpy lit-parameter-evaluator
//===----------------------------------------------------------------------===//
bool CommandLLVMDebugEnable::DoExecute(SBDebugger debugger, char **command,
                                       SBCommandReturnObject &result) {
  size_t count = 0;
  for (char **it = command; it && *it; ++it)
    ++count;

  llvm::DebugFlag = true;
#ifndef NDEBUG
  llvm::setCurrentDebugTypes(const_cast<const char **>(command), count);
#else
  setCurrentDebugTypes(const_cast<const char **>(command), count);
#endif
  result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  return true;
}

//===----------------------------------------------------------------------===//
// CommandLLVMDebugDisable: llvm-debug disable
//===----------------------------------------------------------------------===//
bool CommandLLVMDebugDisable::DoExecute(SBDebugger debugger, char **command,
                                        SBCommandReturnObject &result) {
  llvm::DebugFlag = false;
  result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  return true;
}

void M::KGEN::Mojo::registerLLVMDebugCommands(SBDebugger debugger) {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommand root = interpreter.AddMultiwordCommand(
      "llvm-debug", "Commands used to enable and disable LLVM debug logs");
  root.AddCommand("enable", new CommandLLVMDebugEnable(),
                  "Enable LLVM debug logs for the specified types. If no types "
                  "are specified, all types are enabled.");
  root.AddCommand("disable", new CommandLLVMDebugDisable(),
                  "Disable all LLVM debug logs.");
}
