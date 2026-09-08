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
//
// This file defines debug options that apply to the `mojo debug` command.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD
#define KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Debugger Options
//===----------------------------------------------------------------------===//

def DebuggerOptionGroup : OptionGroup<"Debugger options">, Index<11>;

defm xdebugger: CommandOption<["--", "-"], "X",
  DebuggerOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Passes ARG as an argument to the debugger when the debug session "
  "is launched using the debugger command-line interface. This option can be "
  "specified multiple times. It is ignored when using the RPC mode.">>;

//===----------------------------------------------------------------------===//
// Debug Server Options
//===----------------------------------------------------------------------===//

def DebugServerOptionGroup : OptionGroup<"Debug server options">, Index<12>;

def vscode: Flag<["--", "-"], "vscode">,
  Group<DebugServerOptionGroup>, Index<1>,
  HelpText<"Launches the debug session on VS Code via the Mojo extension.">;

def rpc: Flag<["--", "-"], "rpc">,
  Group<DebugServerOptionGroup>, Index<2>,
  HelpText<"Alias for --vscode.">;

defm terminal: CommandOptionWithValues<
          ["--"], "terminal", DebugServerOptionGroup, Index<3>,
          MetaVarName<"TERMINAL">,
          HelpText<"The type of terminal to use when starting a launch debug "
                   "session.\n\n"
                   "* `console` (default): the debuggee will be launched in the"
                   " default environment for the editor. If using VS Code, this"
                   " will be the Debug Console.\n"
                   "* `dedicated`: the debuggee will be launched in a dedicated"
                   " terminal within the editor.">,
          Values<"console,integrated">>;

defm port: CommandOption<["--", "-"], "port",
  DebugServerOptionGroup, Index<4>, MetaVarName<"PORT">,
  HelpText<"Uses the given PORT to communicate with the RPC debug server. "
           "Defaults to trying all ports from 12355 to 12364 inclusive.">>;

def stopOnEntry: Flag<["--", "-"], "stop-on-entry">,
  Group<DebugServerOptionGroup>, Index<5>,
  HelpText<"Automatically stop after launch.">;

defm initCommand: CommandOption<["--", "-"], "init-command",
  DebugServerOptionGroup, Index<6>, MetaVarName<"INIT_COMMAND">,
  HelpText<"Initialization command executed upon debugger startup. Can be "
           "specified multiple times.">>;

#endif // KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD
