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

include "Debug/DebugDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"
include "../Common/DiagnosticOptions.td"
include "../Common/TargetOptions.td"
include "../Common/StandaloneDebugOptions.td"


//===----------------------------------------------------------------------===//
// Attach Options
//===----------------------------------------------------------------------===//

def AttachOptionGroup : OptionGroup<"Attach options">, Index<2>;

defm pid: CommandOption<["--", "-"], "pid",
  AttachOptionGroup, Index<1>, MetaVarName<"PID">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "PID.">>;

defm process_name: CommandOption<["--", "-"], "process-name",
  AttachOptionGroup, Index<2>, MetaVarName<"NAME">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "name or path.">>;

// As this option is hidden, it needs to exist within a non-empty option group,
// so we put it here just because there's no place better.
def dry_run: Flag<["--", "-"], "dry-run">,
  Group<AttachOptionGroup>, Index<10>,
  Flags<[HelpHidden]>,
  HelpText<"Instead of executing LLDB on the CLI, dumps the command that would "
  "be used to start the LLDB command-line interface. If using the RPC mode, "
  "dumps the payload that would be sent to the RPC server.">;

//===----------------------------------------------------------------------===//
// cuda-gdb options
//===----------------------------------------------------------------------===//

def cudaGDBOptionGroup : OptionGroup<"cuda-gdb options">, Index<4>;

def cudaGDB: Flag<["--", "-"], "cuda-gdb">,
  Group<cudaGDBOptionGroup>, Index<1>,
  HelpText<"Uses cuda-gdb instead of LLDB for debugging. In this mode, it's "
  "possible to step into GPU code, but the CPU debugging experience is "
  "degraded.">;

defm cudaGDBPath: CommandOption<["--", "-"], "cuda-gdb-path",
  cudaGDBOptionGroup, Index<2>, MetaVarName<"CUDA_GDB_PATH">,
  HelpText<"Uses the given CUDA_GDB_PATH instead of looking for cuda-gdb in "
  "the PATH environment variable.">>;

def breakOnLaunch: Flag<["--", "-"], "break-on-launch">,
  Group<cudaGDBOptionGroup>, Index<3>,
  HelpText<"Set the breakOnLaunch option for cuda-gdb.  This makes the "
  "debugger break on the first instruction of every launched kernel.">;
