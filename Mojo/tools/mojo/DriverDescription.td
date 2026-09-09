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
include "Build/BuildDescription.td"
include "Debug/DebugDescription.td"
include "Demangle/DemangleDescription.td"
include "Doc/DocDescription.td"
include "Format/FormatDescription.td"
include "Precompile/PrecompileDescription.td"
include "REPL/REPLDescription.td"
include "Run/RunDescription.td"

def : MojoDriverDescription<
  "The Mojo🔥 command line interface.",
  "The `mojo` CLI provides all the tools you need for Mojo development, such "
  "as commands to run, compile, and precompile Mojo code. A list of all commands "
  "are listed below, and you can learn more about each one by adding the "
  "`--help` option to the command (for example, `mojo precompile --help`).\n"
  "\n"
  "However, you may omit the `run` and `repl` commands. That is, you can run "
  "a Mojo file by simply passing the filename to `mojo`:\n"
  "\n"
  "    mojo hello.mojo\n"
  "\n"
  "And you can start a REPL session by running `mojo` with no commands.\n"
  "\n"
  "You can check your current version with `mojo --version`. "
  "For version information, see all the "
  "[Mojo releases](https://mojolang.org/releases/).",
  [
    UsageInfo<"", "command", /*variadicInput=*/false>,
    UsageInfo<"run-options", "path", /*variadicInput=*/false>,
    UsageInfo<"options", "", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;
