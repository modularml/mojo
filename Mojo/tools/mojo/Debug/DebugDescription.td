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

include "DescriptionBase.td"

def DebugDescription : MojoCommandDescription<
  "debug", "Launches the Mojo debugger using the command-line interface or an "
  "external editor.",
  "This command, which underneath uses the LLDB debugger, or cuda-gdb, offers "
  "four basic debug session modes:\n\n"
  "* Build and debug a Mojo file.\n\n"
  "        mojo debug [options] <file.mojo> [runtime args]\n\n"
  "    Builds the Mojo file at the given path and launches it under the"
  " debugger. Options, which come before the Mojo file, can include any "
  "compilation options expected by the `mojo run`, as well as regular debugging"
  "commands. Runtime args, which come after the Mojo file, are passed "
  "directly to the debuggee upon launch. By default, this mode uses `-O0` and "
  "`--debug-level=full` as compilation options.\n\n"
  "* Debug a precompiled program.\n\n"
  "        mojo debug [options] <program> [runtime args]\n\n"
  "    Launches the program at the given path in the debugger. Options, which "
  "come before the program path, cannot include compilation commands. Runtime "
  "args, which come after the program path, are passed directly to the debuggee"
  " upon launch.\n\n"
  "* Attach to a running process.\n\n"
  "        mojo debug [options] [--pid <pid> | --process-name <process-name>]\n\n"
  "    Attaches to the process specified by pid or name, which can be the full "
  "path of the process' executable. Options other than the process identifier "
  "cannot include compilation options.\n\n"
  "* Start the debugger command-line interface.\n\n"
  "        mojo debug [options]\n\n"
  "    Launches the debugger CLI with support for debugging Mojo programs. This"
  " command only supports LLDB or cuda-gdb options via the `--X` option.\n\n"
  "You can also select one of two interfaces for the debug session:\n\n"
  "* CLI: By default, all debug session modes are launched using the "
  "regular debugger command-line interface.\n\n"
  "* VS Code Debug Server: If you add the `--vscode` option, the debug session "
  "is launched in VS Code via the Mojo extension. VS Code must be running and "
  "the Mojo extension must be enabled. Besides that, the environment"
  " variables and the current working directory of this invocation are "
  "preserved when launching programs in the debugger on VS Code.\n\n"
  "Finally, it is worth mentioning that this debugger can debug programs "
  "written in other standard native languages like Rust, C and C++, as it is "
  "based on LLDB or cuda-gdb.\n\n"
  "Debugger capabilities:\n\n"
  "* LLDB: this is the default debugger and has great support for CPU Mojo "
  "code, but has no support at all for Mojo GPU code.\n\n"
  "* cuda-gdb: this is invoked via the `--cuda-gdb` option and has minimal "
  "support for CPU Mojo code but it has support for GPU Mojo code.",
  [UsageInfo<"debug-options", "", /*variadicInput=*/false>]>, Index<4>;
